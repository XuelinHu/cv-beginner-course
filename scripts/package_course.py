"""只打包明确允许的学生课程文件，不包含飞书私人备份或任何凭据。"""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ['README.md', 'requirements.txt', 'requirements-optional.txt', 'LICENSE',
         'THIRD_PARTY_NOTICES.md', 'course_utils.py', 'industrial_data.py', 'keypoint_utils.py', '.gitignore', '.gitattributes']
FOLDERS = ['notebooks', 'assets', 'reference']
INDUSTRIAL_COURSE = ROOT/'courses'/'industrial-vision-lab'
INDUSTRIAL_EXCLUDED_PARTS = {'.git', '.venv', '__pycache__', '.ipynb_checkpoints',
                             'runtime', 'private', 'cache', 'site-private', 'notebooks', 'executed'}
DOCS = ['CURRICULUM.md', 'SOURCES.md', 'VERIFICATION.md']
SCRIPTS = ['run_notebooks.py', 'package_course.py', 'audit_course.py']


def main():
    destination = ROOT/'dist'/'cv-beginner-course-student.zip'
    destination.parent.mkdir(exist_ok=True)
    paths = [ROOT/f for f in FILES] + [ROOT/'docs'/f for f in DOCS] + [ROOT/'scripts'/f for f in SCRIPTS]
    for folder in FOLDERS:
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.ipynb_checkpoints' not in p.parts)
    if INDUSTRIAL_COURSE.is_dir():
        paths.extend(p for p in INDUSTRIAL_COURSE.rglob('*')
                     if p.is_file()
                     and not INDUSTRIAL_EXCLUDED_PARTS.intersection(p.relative_to(INDUSTRIAL_COURSE).parts)
                     and '.sqlite' not in p.name.lower()
                     and p.suffix.lower() != '.db')
    missing = [str(p) for p in paths if not p.is_file()]
    if missing: raise FileNotFoundError(missing)
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in paths:
            archive.write(p, Path('cv-beginner-course')/p.relative_to(ROOT))
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert not any('feishu-backup' in n or 'feishu-revision' in n or '.git/' in n for n in names)
        assert not any('/runtime/' in n or '/data/private/' in n or '.sqlite' in n.lower() or n.endswith('.db') for n in names)
    print(destination)
    print(f'{len(paths)} 文件，{destination.stat().st_size/1024**2:.1f} MB')


if __name__ == '__main__': main()
