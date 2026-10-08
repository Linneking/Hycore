import io
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

from hier_original_dataset_fetch_v1 import extract_archive, safe_name


class DatasetArchiveChecks(unittest.TestCase):
    def test_escape_absolute_and_windows_paths_rejected(self):
        for name in ('../escape.jpg', '/absolute.jpg', 'a/../../escape.jpg', 'a\\escape.jpg'):
            with self.assertRaises(ValueError):
                safe_name(name)

    def test_valid_zip_bytes_preserved(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            archive = root/'input.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('images/a.jpg', b'original-image-bytes')
            extract_archive(archive, root/'output')
            self.assertEqual((root/'output/images/a.jpg').read_bytes(), b'original-image-bytes')

    def test_tar_link_rejected_before_writing_files(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            archive = root/'input.tgz'
            with tarfile.open(archive, 'w:gz') as tar:
                image = tarfile.TarInfo('images/valid.jpg'); image.size = 4
                tar.addfile(image, io.BytesIO(b'data'))
                link = tarfile.TarInfo('images/link.jpg'); link.type = tarfile.SYMTYPE; link.linkname = '../../escape'
                tar.addfile(link)
            with self.assertRaises(ValueError):
                extract_archive(archive, root/'output')
            self.assertEqual(list((root/'output').rglob('*')), [])


if __name__ == '__main__':
    unittest.main()
