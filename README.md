# 📸 Local Photo Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Development Status](https://img.shields.io/badge/status-alpha-red.svg)](https://github.com/AustinOrphan/local-photo-analyzer)

> **🔒 Privacy-First Photo Organization with Local AI**
> 
> Intelligently analyze, rename, tag, and organize your photos using local LLM models while maintaining complete privacy. No cloud uploads, no data sharing, just smart local photo management.

## ✨ Features

### 🤖 AI-Powered Analysis
- **Local LLM Integration**: Uses Ollama with LLaVA for vision-language understanding
- **Content Recognition**: Automatically identifies objects, scenes, and activities
- **Smart Tagging**: Generates relevant tags based on image content
- **Intelligent Naming**: Creates descriptive filenames from image analysis

### 📁 Smart Organization
- **Date-Based Structure**: Primary organization by YYYY/MM/DD from EXIF data
- **Symbolic Links**: Tag-based categorical access via symbolic links
- **Duplicate Detection**: Identifies and handles duplicate images
- **Batch Processing**: Efficiently processes large photo collections

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

1. **Clone the repository**
   ```bash
   git clone https://github.com/AustinOrphan/local-photo-analyzer.git
   cd local-photo-analyzer
   ```

2. **Set up Python environment**
   ```bash
   # Using Poetry (recommended)
   poetry install
   poetry shell
   
   # Or using pip
   pip install -r requirements.txt
   ```

3. **Install and set up Ollama**
   ```bash
   # Install Ollama (Linux/Mac)
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Pull the LLaVA model for image analysis
   ollama pull llava
   
   # Optional: Pull additional models
   ollama pull llama3.2-vision:90b  # Larger, more accurate model
   ```

4. **Initialize the application**
   ```bash
   python -m photo_analyzer init
   ```

### Basic Usage

```bash
# Analyze a single photo
photo-analyzer analyze /path/to/photo.jpg

# Process a directory of photos
photo-analyzer process /path/to/photos/

# Start the web interface
photo-analyzer serve

# CLI help
photo-analyzer --help
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

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | 🚧 In Progress | Foundation & Setup |
| Phase 2 | 📋 Planned | Core Analysis Engine |
| Phase 3 | 📋 Planned | File Organization System |
| Phase 4 | 📋 Planned | User Interface |
| Phase 5 | 📋 Planned | Advanced Features |

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