#!/bin/bash
# ============================================================
# 服务器连接稳定性 & 写入鲁棒性测试脚本
# 测试项：写入完整性、网络延迟、磁盘I/O
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_FILE="$SCRIPT_DIR/网络测试.txt"
BACKUP_CONTENT=$(cat "$TARGET_FILE" 2>/dev/null || echo "")
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
REPORT_FILE="$SCRIPT_DIR/网络测试报告_${TIMESTAMP}.txt"
TEMP_DIR="$SCRIPT_DIR/.test_tmp"
ROUNDS=3
WRITE_ROUNDS=10

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 初始化
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

log() {
    echo -e "$1" | tee -a "$REPORT_FILE"
}

section() {
    echo "" | tee -a "$REPORT_FILE"
    echo -e "${CYAN}============================================================${NC}" | tee -a "$REPORT_FILE"
    echo -e "${CYAN}  $1${NC}" | tee -a "$REPORT_FILE"
    echo -e "${CYAN}============================================================${NC}" | tee -a "$REPORT_FILE"
}

# ============================================================
# 测试 1: 写入鲁棒性测试
# ============================================================
test_write_robustness() {
    local round=$1
    section "第 ${round} 轮 - 写入鲁棒性测试 (${WRITE_ROUNDS} 次覆写)"

    local pass=0
    local fail=0
    local total_time=0

    for i in $(seq 1 $WRITE_ROUNDS); do
        local write_content="[测试轮次 ${round}-${i}] $(date '+%Y-%m-%d %H:%M:%S.%3N') | UUID=$(uuidgen 2>/dev/null || cat /proc/sys/kernel/random/uuid) | 随机数=$RANDOM"
        local expected_md5=$(printf '%s' "$write_content" | md5sum | awk '{print $1}')

        local start_ns=$(date +%s%N)
        printf '%s' "$write_content" > "$TARGET_FILE"
        local write_rc=$?
        sync
        local end_ns=$(date +%s%N)

        local actual_md5=$(md5sum "$TARGET_FILE" | awk '{print $1}')
        local elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))

        if [ "$write_rc" -eq 0 ] && [ "$expected_md5" = "$actual_md5" ]; then
            log "  ${GREEN}[PASS]${NC} 第 ${i} 次写入成功 | 耗时: ${elapsed_ms}ms | md5: ${actual_md5:0:12}..."
            pass=$((pass + 1))
        else
            log "  ${RED}[FAIL]${NC} 第 ${i} 次写入失败 | 耗时: ${elapsed_ms}ms | 期望md5: ${expected_md5:0:12}... | 实际md5: ${actual_md5:0:12}..."
            fail=$((fail + 1))
        fi
        total_time=$((total_time + elapsed_ms))
    done

    log ""
    log "  写入鲁棒性汇总: ${GREEN}通过 ${pass}${NC} / ${RED}失败 ${fail}${NC}"
    if [ $WRITE_ROUNDS -gt 0 ]; then
        log "  平均写入耗时: $(( total_time / WRITE_ROUNDS ))ms"
    fi

    # 返回失败数
    return $fail
}

# ============================================================
# 测试 2: 网络连通性测试
# ============================================================
test_network() {
    local round=$1
    section "第 ${round} 轮 - 网络连通性测试"

    # Ping 8.8.8.8
    log "  [1] Ping 8.8.8.8 (5 包)..."
    local ping_result
    ping_result=$(ping -c 5 -W 3 8.8.8.8 2>&1) && local ping_ok=0 || local ping_ok=1
    echo "$ping_result" | grep -E "packet loss|rtt|avg" | while read line; do
        log "      $line"
    done
    if [ "$ping_ok" -eq 0 ]; then
        log "  ${GREEN}[PASS]${NC} Ping 8.8.8.8 成功"
    else
        log "  ${RED}[FAIL]${NC} Ping 8.8.8.8 失败"
    fi

    # Ping github.com
    log "  [2] Ping github.com (5 包)..."
    ping_result=$(ping -c 5 -W 3 github.com 2>&1) && local ping_ok2=0 || local ping_ok2=1
    echo "$ping_result" | grep -E "packet loss|rtt|avg|unknown" | while read line; do
        log "      $line"
    done
    if [ "$ping_ok2" -eq 0 ]; then
        log "  ${GREEN}[PASS]${NC} Ping github.com 成功"
    else
        log "  ${YELLOW}[WARN]${NC} Ping github.com 失败（可能被墙或DNS问题）"
    fi
}

# ============================================================
# 测试 3: 磁盘 I/O 测试
# ============================================================
test_disk_io() {
    local round=$1
    section "第 ${round} 轮 - 磁盘 I/O 测试"

    local test_file="$TEMP_DIR/io_test_${round}.bin"
    local size_mb=10

    log "  [1] 顺序写入 ${size_mb}MB 文件..."
    local dd_result
    dd_result=$(dd if=/dev/zero of="$test_file" bs=1M count=$size_mb conv=fsync 2>&1) && local dd_ok=0 || local dd_ok=1
    echo "$dd_result" | while read line; do
        log "      $line"
    done

    if [ "$dd_ok" -eq 0 ]; then
        log "  ${GREEN}[PASS]${NC} 磁盘写入成功"

        # 校验
        log "  [2] 校验文件完整性..."
        local actual_size=$(stat -c%s "$test_file" 2>/dev/null || stat -f%z "$test_file" 2>/dev/null)
        local expected_size=$((size_mb * 1024 * 1024))
        if [ "$actual_size" -eq "$expected_size" ]; then
            log "  ${GREEN}[PASS]${NC} 文件大小校验通过 (${actual_size} bytes = ${size_mb}MB)"
        else
            log "  ${RED}[FAIL]${NC} 文件大小不匹配: 期望 ${expected_size}, 实际 ${actual_size}"
        fi
    else
        log "  ${RED}[FAIL]${NC} 磁盘写入失败"
    fi

    # 清理
    rm -f "$test_file"
    log "  [3] 临时文件已清理"
}

# ============================================================
# 主流程
# ============================================================
main() {
    log "============================================================"
    log "  服务器连接稳定性 & 写入鲁棒性测试报告"
    log "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    log "  主机名: $(hostname)"
    log "  工作目录: $SCRIPT_DIR"
    log "  测试轮数: ${ROUNDS} 轮"
    log "============================================================"

    local total_write_pass=0
    local total_write_fail=0

    for round in $(seq 1 $ROUNDS); do
        log ""
        log "############################################################"
        log "  >>> 开始第 ${round}/${ROUNDS} 轮测试 <<<"
        log "############################################################"

        test_write_robustness $round
        total_write_pass=$((total_write_pass + WRITE_ROUNDS - $?))
        total_write_fail=$((total_write_fail + $?))

        test_network $round
        test_disk_io $round

        if [ $round -lt $ROUNDS ]; then
            log ""
            log "  等待 3 秒后开始下一轮..."
            sleep 3
        fi
    done

    # 恢复原始内容
    echo "$BACKUP_CONTENT" > "$TARGET_FILE"

    # 清理
    rm -rf "$TEMP_DIR"

    # 总结
    section "测试总结"
    log "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
    log "  总写入次数: $((ROUNDS * WRITE_ROUNDS))"
    log "  写入通过: ${total_write_pass}"
    log "  写入失败: ${total_write_fail}"
    if [ "$total_write_fail" -eq 0 ]; then
        log "  ${GREEN}结论: 所有写入测试通过，连接稳定，写入鲁棒！${NC}"
    else
        log "  ${RED}结论: 存在 ${total_write_fail} 次写入失败，请检查服务器状态！${NC}"
    fi

    log ""
    log "  报告已保存至: $REPORT_FILE"
    log "  原始文件已恢复: $TARGET_FILE"
    log "============================================================"
}

main