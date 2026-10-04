"""V6-H20 proxy top-four visualization using a complete same-epoch checkpoint.

Requires --expected-epoch, including when the source is best.pth. The shared
V5 visualization implementation keeps clean features, FP64 raw Poincare
retrieval, seeded eligible-proxy selection, and offline PNG/HTML output.
"""

if __package__:
    from .visualize_v5_proxy_neighbors import main
else:
    from visualize_v5_proxy_neighbors import main


if __name__ == "__main__":
    main(version="v6")
