# migrate_mantra.ps1
# PowerShell script to restructure MANTRA repository into modular architecture
# Run in project root

# 1️⃣ Create folder structure
New-Item -ItemType Directory -Force -Path "mantra/domain"
New-Item -ItemType Directory -Force -Path "mantra/application"
New-Item -ItemType Directory -Force -Path "mantra/fingerprinting/midi"
New-Item -ItemType Directory -Force -Path "mantra/fingerprinting/audio"
New-Item -ItemType Directory -Force -Path "mantra/similarity"
New-Item -ItemType Directory -Force -Path "mantra/index"
New-Item -ItemType Directory -Force -Path "mantra/storage"
New-Item -ItemType Directory -Force -Path "mantra/interfaces/api"
New-Item -ItemType Directory -Force -Path "mantra/interfaces/cli"
New-Item -ItemType Directory -Force -Path "mantra/interfaces/plugin_gateway"

# 2️⃣ Move working files
Move-Item -Force "builder/fingerprint_builder.py" "mantra/fingerprinting/midi/"
Move-Item -Force "core/fingerprint.py" "mantra/fingerprinting/midi/"
Move-Item -Force "core/similarity.py" "mantra/similarity/"
Move-Item -Force "indexing/similarity_engine.py" "mantra/index/"
Move-Item -Force "persistence/sqlite_store.py" "mantra/storage/"
Move-Item -Force "app/main.py" "mantra/interfaces/api/"

# 3️⃣ Fix imports in moved files (basic)
Get-ChildItem -Recurse mantra -Filter *.py | ForEach-Object {
    (Get-Content $_.FullName) -replace 'from core\.fingerprint', 'from mantra.fingerprinting.midi.fingerprint' `
                               -replace 'from core\.similarity', 'from mantra.similarity.similarity' `
                               -replace 'from builder\.fingerprint_builder', 'from mantra.fingerprinting.midi.fingerprint_builder' |
    Set-Content $_.FullName
}

# 4️⃣ Run tests to verify
pytest -q --disable-warnings

# 5️⃣ Commit and push
git add .
git commit -m "Reorganized MANTRA into modular architecture; migrated end-to-end index+search with fixed imports"
git push -u origin main