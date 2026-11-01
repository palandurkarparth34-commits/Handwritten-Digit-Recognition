# Applying this rebuild to the GitHub repository

The GitHub connection used for this session can read the repository but does not have permission to create branches or push files. The original `main` branch has not been changed.

After downloading and extracting this rebuild bundle, open a terminal in the extracted folder and run:

```bash
./apply-rebuild.sh /absolute/path/to/Handwritten-Digit-Recognition
```

The script checks that the destination is the intended Git repository, refuses to overwrite an existing `rebuild` branch, requires a clean working tree, fetches the latest `main`, creates `rebuild`, replaces the old app, removes the no-longer-used `src/` and `data/` code plus the TensorFlow requirements file, commits the result, and pushes `origin/rebuild`. It keeps the existing `LICENSE` and historical `docs/` folder.

The script does not merge into `main` or delete the remote `main` branch.

If you prefer to do it manually:

```bash
git clone https://github.com/palandurkarparth34-commits/Handwritten-Digit-Recognition.git
cd Handwritten-Digit-Recognition
git switch -c rebuild origin/main
# Copy app.py, README.md, requirements*.txt, .gitignore, templates/, and tests/
# from this bundle into this repository, replacing the matching files.
rm -rf src data
rm -f requirements-cnn.txt
git add -A
git commit -m "Rebuild digit recognition app with polished UI and reliable inference"
git push -u origin rebuild
```
