# GitHub Actions Workflows

This directory contains the GitHub Actions workflows for building the Contract Intelligence Platform Electron app.

## Workflows

### `build-electron-simple.yml` ✅ **ACTIVE**
- **Purpose**: Simplified, reliable build process for Windows and macOS installers
- **Approach**: Uses system Python with pip-installed dependencies
- **Benefits**: 
  - Much simpler and more reliable
  - Easier to maintain and debug
  - Faster builds with better error handling
  - Uses proven community patterns

### `build-electron.yml` ⚠️ **BACKUP**
- **Purpose**: Complex build process with portable Python bundling
- **Status**: Renamed to backup, contains the original complex approach
- **Issues**: 
  - Complex portable Python setup prone to failures
  - Multiple shell script compatibility issues
  - Hard to maintain and debug

## Usage

### Automatic Builds
- **Tag Push**: Push a git tag (e.g., `v1.5.6`) to trigger automatic builds for both platforms
- **Manual Trigger**: Use GitHub Actions "Run workflow" button for manual builds

### Requirements
The simplified workflow requires:
1. **Node.js dependencies**: Automatically installed via `npm ci`
2. **Python dependencies**: Automatically installed via `pip install -r requirements.txt`
3. **Build assets**: Icons must be present in `electron-app/build/`
   - `icon.ico` for Windows builds
   - `icon.icns` for macOS builds

### Output
- **Artifacts**: Uploaded to GitHub Actions for 90 days
- **Releases**: Automatic GitHub releases for tag builds
- **Files**: 
  - Windows: `Contract Intelligence Platform Setup *.exe`
  - macOS: `Contract Intelligence Platform-*.dmg`

## Migration Notes

The new simplified approach:
- ✅ **Removes complex portable Python setup** - uses system Python instead
- ✅ **Simplifies shell scripts** - better cross-platform compatibility  
- ✅ **Improves error handling** - clearer failure messages
- ✅ **Faster builds** - fewer steps, better caching
- ⚠️ **Requires Python on target systems** - users must have Python 3.12+ installed

This trade-off significantly improves build reliability while requiring users to have Python installed (which is reasonable for a development-focused tool).
