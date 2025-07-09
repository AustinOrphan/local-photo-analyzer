# 📸 Local Photo Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Development Status](https://img.shields.io/badge/status-beta-orange.svg)](https://github.com/AustinOrphan/local-photo-analyzer)

> **🔒 Privacy-First Photo Organization with Local AI**
> 
> Intelligently analyze, rename, tag, and organize your photos using local LLM models while maintaining complete privacy. No cloud uploads, no data sharing, just smart local photo management.

## ✨ Features

### 🤖 AI-Powered Analysis
- **Ensemble Models**: Multiple LLM models working together for higher accuracy
- **Advanced Content Recognition**: Objects, scenes, activities, and artistic elements
- **Quality Assessment**: Technical quality metrics and composition analysis
- **Smart Tagging**: Context-aware tags with confidence scoring
- **Intelligent Naming**: Descriptive filenames from comprehensive analysis

### 📁 Smart Organization
- **Date-Based Structure**: Primary organization by YYYY/MM/DD from EXIF data
- **Symbolic Links**: Tag-based categorical access via symbolic links
- **Advanced Duplicate Detection**: Perceptual hashing with exact, near, and similar matching
- **Batch Processing**: Scalable processing with progress tracking and error recovery
- **Scene Analysis**: Color extraction and composition scoring

### 🔐 Privacy & Security
- **100% Local Processing**: All analysis happens on your machine
- **No Cloud APIs**: No data ever leaves your computer
- **Secure Database**: Local SQLite/PostgreSQL storage
- **Audit Logging**: Track all file operations for transparency

### 🎯 User Control
- **Manual Override**: Review and correct AI decisions
- **Custom Tags**: Add your own tags and categories
- **Flexible Rules**: Customize organization patterns
- **Rollback Support**: Undo changes safely

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- [Ollama](https://ollama.com/) for local LLM models
- 4GB+ RAM (8GB+ recommended for large collections)
- Modern CPU (GPU optional but recommended)

### Installation

> **✅ The system is fully functional!** The core analysis pipeline, CLI commands, and web interface are working. Follow these steps to get started.

1. **Clone the repository**
   ```bash
   git clone https://github.com/AustinOrphan/local-photo-analyzer.git
   cd local-photo-analyzer
   ```

2. **Set up Python environment with Poetry (recommended)**
   ```bash
   # Using Poetry
   poetry install
   poetry shell
   ```

3. **Alternative: Install with pip**
   ```bash
   # Install Python dependencies
   pip install -r requirements.txt
   
   # Install the package in editable mode (important!)
   pip install -e .
   ```

4. **Install and set up Ollama**
   ```bash
   # Install Ollama (Linux/Mac)
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Start Ollama service
   ollama serve
   
   # Pull the LLaVA model for image analysis (in another terminal)
   ollama pull llava
   
   # Optional: Pull additional models
   ollama pull llama3.2:latest     # Alternative text model
   ollama pull qwen2.5-coder:latest  # Code analysis model
   ```

5. **Initialize the application**
   ```bash
   # Initialize database and configuration
   python -m src.photo_analyzer.cli.main init
   
   # Check system status
   python -m src.photo_analyzer.cli.main status
   ```

## 🛠️ Troubleshooting

### "No module named photo_analyzer" Error

If you get this error, the package may not be properly installed. Try these solutions:

1. **Use the full module path** (current working method):
   ```bash
   python -m src.photo_analyzer.cli.main init
   ```

2. **Install in editable mode**:
   ```bash
   pip install -e .
   ```

3. **Alternative: Add to Python path**:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   python -m photo_analyzer.cli.main init
   ```

### Ollama Connection Issues

If photo analysis fails:
1. Ensure Ollama is running: `ollama serve`
2. Test the model: `ollama run llava "describe this image" < test_image.jpg`
3. Check the Ollama API endpoint in your config

## 💡 Quick Usage Examples

### CLI Commands (Working!)

```bash
# Analyze a single photo
python -m src.photo_analyzer.cli.main analyze sample_photos/sample_landscape.jpg

# Analyze all photos in a directory
python -m src.photo_analyzer.cli.main analyze /path/to/photos/

# Search for photos by content
python -m src.photo_analyzer.cli.main search "house"
python -m src.photo_analyzer.cli.main search "nature"

# Preview smart renaming (dry run)
python -m src.photo_analyzer.cli.main rename /path/to/photos/ --dry-run

# Apply smart renaming
python -m src.photo_analyzer.cli.main rename /path/to/photos/

# Organize photos by date with symlinks
python -m src.photo_analyzer.cli.main organize /path/to/photos/ /organized/output/

# Check system status
python -m src.photo_analyzer.cli.main status

# Get help for any command
python -m src.photo_analyzer.cli.main --help
python -m src.photo_analyzer.cli.main analyze --help
```

### Web Interface

```bash
# Start the web server using CLI command (recommended)
python -m src.photo_analyzer.cli.main serve --host 0.0.0.0 --port 8000

# Alternative: Start using the dedicated server script
python -m src.photo_analyzer.web.server --host 0.0.0.0 --port 8000

# Development mode with auto-reload
python -m src.photo_analyzer.cli.main serve --reload

# Then open http://localhost:8000 in your browser
# API documentation available at: http://localhost:8000/docs
```

### Real Example Output

```bash
$ python -m src.photo_analyzer.cli.main analyze sample_photos/sample_landscape.jpg

Found 1 image files to analyze
  Analyzing photos... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:06

                             Photo Analysis Results                             
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ File                 ┃ Description           ┃ Tags         ┃ Confidence ┃ Status    ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ sample_landscape.jpg │ Digital illustration  │ nature,      │ 0.85       │ ✓ Success │
│                      │ of a rural scene...   │ residential, │            │           │
│                      │                       │ house, trees │            │           │
└──────────────────────┴───────────────────────┴──────────────┴────────────┴───────────┘
```

## 📖 Documentation

- **[📋 Roadmap](ROADMAP.md)** - Development phases and timeline
- **[🛠 Installation Guide](docs/installation.md)** - Detailed setup instructions
- **[💡 Usage Guide](docs/usage.md)** - How to use all features
- **[🔧 Configuration](docs/configuration.md)** - Customization options
- **[🤝 Contributing](CONTRIBUTING.md)** - How to contribute
- **[❓ FAQ](docs/faq.md)** - Frequently asked questions

## 🏗 Architecture

### Technology Stack
- **Backend**: Python 3.9+, FastAPI, SQLAlchemy
- **Local AI**: Ollama, LLaVA, CLIP
- **Database**: SQLite (default) / PostgreSQL
- **Frontend**: React, TypeScript
- **CLI**: Click framework

### Project Structure
```
local-photo-analyzer/
├── src/
│   ├── analyzer/          # Core analysis engine
│   ├── organizer/         # File organization logic
│   ├── models/            # Data models and database
│   ├── api/               # REST API endpoints
│   ├── cli/               # Command line interface
│   └── utils/             # Utilities and helpers
├── web/                   # Frontend application
├── tests/                 # Test suite
├── docs/                  # Documentation
└── configs/               # Configuration files
```

## 🔄 How It Works

1. **📷 Photo Ingestion**: Scan directories for image files
2. **🔍 EXIF Extraction**: Extract date, location, and camera metadata
3. **🤖 AI Analysis**: Use local LLM to analyze image content
4. **🏷 Tag Generation**: Create relevant tags from analysis
5. **📅 Date Organization**: Move files to YYYY/MM/DD structure
6. **🔗 Symbolic Linking**: Create tag-based categorical access
7. **💾 Metadata Storage**: Save results to local database

## 📊 Development Status

| Phase | Status | Description | Features |
|-------|--------|-------------|----------|
| Phase 1 | ✅ **Complete** | Foundation & Core Infrastructure | Database, config, logging, CLI framework |
| Phase 2 | ✅ **Complete** | Analysis Engine & Processing Pipeline | LLM integration, image analysis, tag extraction |
| Phase 3 | ✅ **Complete** | Web Interface & Basic API | FastAPI app, photo upload, analysis endpoint |
| Phase 4 | ✅ **Complete** | CLI Commands & Search | All CLI commands working, search functionality |
| Phase 5 | 📋 **Future** | Advanced Features & Optimization | Performance tuning, advanced organization |

### ✅ Currently Working Features:
- **Image Analysis**: LLaVA-powered description and tag generation
- **CLI Commands**: `analyze`, `search`, `rename`, `organize`, `status`
- **Database Storage**: SQLite with proper relationship management
- **Web Interface**: Full FastAPI server with photo upload, analysis, search, and management endpoints
- **Web Server**: Production-ready server with CLI and standalone startup options
- **Smart Renaming**: AI-generated descriptive filenames
- **Content Search**: Search photos by description, tags, or filename
- **EXIF Processing**: Metadata extraction and date parsing

See the [detailed roadmap](ROADMAP.md) for more information.

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Ways to Contribute
- 🐛 **Bug Reports**: Found an issue? [Create an issue](https://github.com/AustinOrphan/local-photo-analyzer/issues)
- 💡 **Feature Requests**: Have an idea? [Start a discussion](https://github.com/AustinOrphan/local-photo-analyzer/discussions)
- 📝 **Documentation**: Improve our docs
- 🧪 **Testing**: Help test on different platforms
- 💻 **Code**: Submit pull requests

### Development Setup
```bash
# Clone and setup for development
git clone https://github.com/AustinOrphan/local-photo-analyzer.git
cd local-photo-analyzer

# Install development dependencies
poetry install --dev

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest

# Start development server
python -m photo_analyzer serve --dev
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 🔒 Privacy & Security

### Privacy Guarantees
- ✅ **No Cloud Processing**: All AI analysis happens locally
- ✅ **No Data Collection**: We don't collect any user data
- ✅ **No Network Requests**: No external API calls for photo analysis
- ✅ **Local Storage**: All data stays on your machine
- ✅ **Open Source**: Fully auditable codebase

### Security Features
- 🔐 Encrypted local database storage
- 🛡️ Safe file operations with atomic moves
- 📝 Comprehensive audit logging
- 🔄 Rollback capabilities for all operations
- 🧪 Sandbox mode for testing

## 🎯 Use Cases

### Personal Photo Management
- Organize family photos by date and content
- Find photos by searching for objects or scenes
- Create themed collections automatically
- Clean up messy photo libraries

### Professional Photography
- Organize client photo shoots efficiently
- Tag photos by equipment, settings, or style
- Create portfolio collections automatically
- Maintain organized archives

### Research & Documentation
- Organize research photos with content-based tagging
- Create searchable visual databases
- Document processes with automatic categorization
- Archive visual evidence systematically

## 📈 Performance

### Benchmarks (Preliminary)
- **Analysis Speed**: ~10-30 seconds per image (CPU)
- **Accuracy**: 85%+ tag accuracy for common objects
- **Memory Usage**: ~2GB for 1000 image collection
- **Storage**: ~10MB database for 10,000 images

*Performance varies based on hardware and model configuration*

## 🔧 Configuration

### Basic Configuration
```yaml
# config.yaml
models:
  primary: "llava"
  fallback: "llama3.2-vision"

organization:
  date_format: "YYYY/MM/DD"
  duplicate_handling: "smart_merge"
  
analysis:
  confidence_threshold: 0.7
  max_tags_per_image: 10
```

See [configuration docs](docs/configuration.md) for all options.

## 🐛 Troubleshooting

### Common Issues

**Ollama model not found**
```bash
# Install the required model
ollama pull llava
```

**Permission errors**
```bash
# Check file permissions
ls -la /path/to/photos/
```

**Out of memory errors**
```bash
# Use smaller model or increase batch size
photo-analyzer process --batch-size 5 /path/to/photos/
```

See [FAQ](docs/faq.md) for more troubleshooting tips.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.com/) - Local LLM runtime
- [LLaVA](https://github.com/haotian-liu/LLaVA) - Vision-language model
- [CLIP](https://github.com/openai/CLIP) - Image-text understanding
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [React](https://reactjs.org/) - Frontend framework

## 📞 Support

- 📖 **Documentation**: Check our [docs](docs/)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/AustinOrphan/local-photo-analyzer/discussions)
- 🐛 **Issues**: [GitHub Issues](https://github.com/AustinOrphan/local-photo-analyzer/issues)
- 📧 **Contact**: [Open an issue](https://github.com/AustinOrphan/local-photo-analyzer/issues/new)

---

<div align="center">

**Built with ❤️ for privacy-conscious photo enthusiasts**

[⭐ Star this repo](https://github.com/AustinOrphan/local-photo-analyzer) if you find it useful!

</div>