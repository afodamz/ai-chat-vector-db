# 🤖 AI Chat with Vector Database - Multi-Model Support

A powerful AI chat application that supports multiple AI models (Gemini, ChatGPT, Claude, Llama, GPT4All, Perplexity AI) with document context, web search, and image processing capabilities. Features a dedicated database manager for efficient vector database operations.

## 🚀 **Quick Start**

### **1. Setup Environment**
```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file with your API keys
cp .env.example .env  # Edit with your keys
```

### **2. Build Your Knowledge Database**
```bash
# Build database from your documents
python db_manager.py --create

# Check database status
python db_manager.py --status
```

### **3. Start Chatting Instantly**
```bash
# Chat with Gemini (default)
python main.py --model gemini

# Chat with ChatGPT
python main.py --model chatgpt

# Chat with Claude
python main.py --model claude
```

## 📁 **Documents Folder**

The `documents/` folder is where you place all your files for the AI to learn from. Simply add your files here and the system will automatically process them.

### **Supported File Types**
- **📄 Text**: `.txt` files
- **📊 Excel**: `.xlsx`, `.xls` files (all sheets processed)
- **📈 CSV**: `.csv` files
- **📝 Word**: `.doc`, `.docx` files
- **📕 PDF**: `.pdf` files
- **🖼️ Images**: `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.tiff`, `.webp`, `.svg`

### **How to Use**
1. **Add your files** to the `documents/` folder
2. **Run database creation**: `python db_manager.py --create`
3. **Start chatting**: `python main.py --model gemini`

### **Example Structure**
```
documents/
├── reports/
│   ├── annual_report.pdf
│   └── quarterly_data.xlsx
├── notes/
│   ├── meeting_notes.docx
│   └── project_ideas.txt
├── images/
│   ├── diagrams.png
│   └── charts.svg
└── data/
    ├── customer_data.csv
    └── sales_data.xlsx
```

## 🗄️ **Database Manager (`db_manager.py`)**

The `db_manager.py` script provides comprehensive database management capabilities, allowing you to create, update, backup, and restore your vector database efficiently.

### **Core Commands**

#### **Database Creation & Management**
```bash
# Create new database from documents
python db_manager.py --create

# Recreate database (clears and rebuilds)
python db_manager.py --recreate

# Update database with new documents
python db_manager.py --update

# Clear entire database
python db_manager.py --clear
```

#### **Database Information**
```bash
# Show database status and statistics
python db_manager.py --status

# Custom database path
python db_manager.py --status --db-path ./my_database

# Custom documents directory
python db_manager.py --create --documents-dir ./my_docs
```

#### **Backup & Restore**
```bash
# Create database backup
python db_manager.py --backup

# Restore from backup
python db_manager.py --restore ./chroma_backup_20241201_143022

# Force operations without confirmation
python db_manager.py --recreate --force
```

### **Advanced Options**
```bash
# Custom paths
python db_manager.py --create --db-path ./custom_db --documents-dir ./custom_docs

# Force operations
python db_manager.py --clear --force
python db_manager.py --recreate --force
```

### **Database Manager Features**
- **🔄 Smart Updates**: Only processes changed documents
- **💾 Automatic Backups**: Timestamped backup creation
- **🔍 Status Monitoring**: Detailed database information
- **⚡ Fast Operations**: Optimized for large document collections
- **🛡️ Safety Checks**: Confirmation prompts for destructive operations

## 💬 **AI Chat (`main.py`)**

### **Available Models**
| Model | Command | Description |
|-------|---------|-------------|
| **Gemini** | `--model gemini` | Google's AI (default, free) |
| **ChatGPT** | `--model chatgpt` | OpenAI's GPT models |
| **Claude** | `--model claude` | Anthropic's Claude |
| **Llama** | `--model llama` | Meta's Llama (local via Ollama) |
| **GPT4All** | `--model gpt4all` | Local GPT4All (via Ollama) |
| **Perplexity** | `--model perplexity` | Perplexity AI |

### **Basic Usage**
```bash
# Default Gemini model
python main.py

# Specific model
python main.py --model chatgpt

# Custom settings
python main.py --model claude --temperature 0.8 --max-tokens 2000

# Disable web search for faster responses
python main.py --model gemini --no-web-search
```

### **Advanced Options**
```bash
# Temperature control (creativity)
python main.py --temperature 0.3    # More focused
python main.py --temperature 0.9    # More creative

# Token limits
python main.py --max-tokens 1000    # Shorter responses
python main.py --max-tokens 4000    # Longer responses

# Chunk settings (affects document processing)
python main.py --chunk-size 1500 --chunk-overlap 300

# Verbose web search detection
python main.py --verbose-web-search
```

### **Chat Commands**
Once in chat mode, you can use these commands:
```
help              - Show all available commands
clear             - Clear chat history
clear db          - Manage database (clear, reload, info)
search image <desc> - Find images by description
image summary     - Show image database stats
process images <dir> - Process all images in directory
history           - Show conversation summary
voice mode        - Switch to voice interaction
speak             - Text-to-speech mode
quit/exit         - End conversation
```

## 🔧 **Installation**

### **1. Clone and Setup**
```bash
git clone <your-repo>
cd langchain
pip install -r requirements.txt
```

### **2. Environment Variables**
Create a `.env` file with your API keys:
```bash
# Google Gemini (Required for gemini model)
GEMINI_API_KEY=your_gemini_api_key_here

# OpenAI ChatGPT (Required for chatgpt model)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Claude (Required for claude model)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Ollama (Required for llama and gpt4all models)
OLLAMA_API_KEY=local_ollama

# Perplexity AI (Required for perplexity model)
PERPLEXITY_API_KEY=your_perplexity_api_key_here
```

### **3. Get API Keys**
- **Gemini**: [Google AI Studio](https://makersuite.google.com/app/apikey)
- **ChatGPT**: [OpenAI Platform](https://platform.openai.com/api-keys)
- **Claude**: [Anthropic Console](https://console.anthropic.com/)
- **Perplexity**: [Perplexity AI](https://www.perplexity.ai/settings/api)
- **Ollama**: [Install Ollama](https://ollama.ai/) for local models

## 📁 **Project Structure**
```
langchain/
├── main.py                    # AI chat interface
├── db_manager.py              # Database management tool
├── documents/                 # Your documents folder (add files here)
│   ├── reports/              # Example: PDF reports
│   ├── data/                 # Example: Excel/CSV files
│   ├── notes/                # Example: Word documents
│   └── images/               # Example: Images and diagrams
├── chroma_data/              # Vector database (auto-created)
├── docker-compose.yml        # Docker setup for ChromaDB
├── .env                      # API keys (create this)
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## 🎯 **Workflow Examples**

### **First Time Setup**
```bash
# 1. Add documents to documents/ folder
# 2. Create database
python db_manager.py --create

# 3. Check database status
python db_manager.py --status

# 4. Start chatting
python main.py --model gemini
```

### **Adding New Documents**
```bash
# 1. Add new files to documents/ folder
# 2. Update database
python db_manager.py --update

# 3. Check updated status
python db_manager.py --status

# 4. Chat with updated knowledge
python main.py --model gemini
```

### **Database Maintenance**
```bash
# Create backup before major changes
python db_manager.py --backup

# Recreate database if needed
python db_manager.py --recreate --force

# Restore from backup if needed
python db_manager.py --restore ./chroma_backup_20241201_143022 --force
```

### **Model Switching**
```bash
# Try different models
python main.py --model chatgpt
python main.py --model claude
python main.py --model llama
```

## 🔍 **Features**

### **Core Capabilities**
- **Multi-Model AI**: Switch between 6+ AI models
- **Document Context**: Chat about your documents
- **Web Search**: Intelligent web search integration
- **Image Processing**: CLIP embeddings for images
- **Vector Database**: ChromaDB for fast retrieval
- **Smart Chunking**: Intelligent document splitting
- **Database Management**: Comprehensive DB operations

### **AI Models Supported**
- **Google Gemini**: Free, powerful, fast
- **OpenAI ChatGPT**: Industry standard
- **Anthropic Claude**: Safety-focused
- **Meta Llama**: Open source, local
- **GPT4All**: Lightweight local model
- **Perplexity AI**: Web-connected AI

### **Document Processing**
- **Excel**: Multiple sheets, column analysis
- **CSV**: Structured data processing
- **Word**: Document content extraction
- **PDF**: Text and layout preservation
- **Images**: CLIP embeddings for visual search
- **SVG**: Vector graphics support

### **Database Management**
- **Create**: Build new databases from documents
- **Update**: Incremental updates for new documents
- **Backup**: Timestamped database backups
- **Restore**: Restore from previous backups
- **Status**: Detailed database information
- **Clear**: Complete database reset

## 🚨 **Troubleshooting**

### **Common Issues**

#### **"No database found"**
```bash
# Create database first
python db_manager.py --create
```

#### **"API Key not found"**
- Check `.env` file exists
- Verify API key variable names
- Restart terminal after creating `.env`

#### **"CLIP not available"**
- Normal if image packages not installed
- Chat still works with text documents
- Install: `pip install transformers torch torchvision cairosvg`

#### **"Import Error"**
- Install requirements: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

#### **Database Issues**
```bash
# Check database status
python db_manager.py --status

# Recreate if corrupted
python db_manager.py --recreate --force

# Restore from backup
python db_manager.py --restore ./backup_path --force
```

### **Performance Tips**
1. **Build database once**, chat often
2. **Use `--update`** for incremental updates
3. **Disable web search** with `--no-web-search` for faster responses
4. **Monitor database size** with `db_manager.py --status`
5. **Create regular backups** with `db_manager.py --backup`
6. **Clear database** only when necessary

## 💡 **Pro Tips**

1. **Start with Gemini** - It's free and works well
2. **Use local models** (Ollama) for privacy-sensitive data
3. **Organize documents** in subfolders for better structure
4. **Monitor API usage** to avoid unexpected charges
5. **Create regular backups** of your database
6. **Use `--force` flag** for automated scripts
7. **Check database status** regularly with `--status`

## 🐳 **Docker Support**

The project includes Docker Compose configuration for ChromaDB:

```bash
# Start ChromaDB server
docker-compose up -d

# Stop ChromaDB server
docker-compose down
```

This is useful for:
- **Production deployments**
- **Multiple users**
- **Persistent database storage**
- **Remote database access**

## 🔄 **Migration Guide**

If you were using an older version:
1. **Your existing database** will still work
2. **Run `python db_manager.py --status`** to check status
3. **Use `python main.py`** for fast chat
4. **Old database** will be automatically detected
5. **Create backups** before major updates

## 📊 **Example Use Cases**

### **Business Intelligence**
```bash
# Add business reports and data
# documents/business/
#   ├── sales_data.xlsx
#   ├── market_analysis.pdf
#   └── quarterly_reports.docx

python db_manager.py --create
python main.py --model gemini
# Ask: "What are the key trends in our sales data?"
```

### **Research Assistant**
```bash
# Add research papers and notes
# documents/research/
#   ├── papers/
#   ├── notes/
#   └── data/

python db_manager.py --create
python main.py --model claude
# Ask: "Summarize the main findings from the research papers"
```

### **Personal Knowledge Base**
```bash
# Add personal documents
# documents/personal/
#   ├── notes/
#   ├── images/
#   └── documents/

python db_manager.py --create
python main.py --model gemini
# Ask: "What did I learn about machine learning last month?"
```

## 🤝 **Contributing**

Feel free to contribute improvements:
- Add new AI model support
- Enhance document processing
- Improve database management
- Add new features
- Optimize performance

## 📚 **Resources**

- **LangChain**: [Documentation](https://python.langchain.com/)
- **ChromaDB**: [Vector Database](https://www.trychroma.com/)
- **CLIP**: [Image Understanding](https://openai.com/research/clip)
- **Ollama**: [Local AI Models](https://ollama.ai/)
- **Docker**: [Container Platform](https://www.docker.com/)

---

## 🎉 **Get Started Now!**

**1. Add your documents:**
```bash
# Add files to documents/ folder
mkdir -p documents/{reports,data,notes,images}
```

**2. Build your knowledge database:**
```bash
python db_manager.py --create
```

**3. Start chatting instantly:**
```bash
python main.py --model gemini
```

**4. Enjoy fast, intelligent AI conversations with your documents!**

---

*Built with ❤️ using LangChain, ChromaDB, and modern AI models*