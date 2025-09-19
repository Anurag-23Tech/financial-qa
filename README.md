# 📊 Financial Document Q&A System

This project allows users to **upload financial documents (PDF/Excel)** and ask natural language questions about them.  
It extracts tables and text, processes them, and uses **Ollama (local LLM)** for reasoning.  

Hybrid approach:
- ✅ Direct lookup (fast answers for exact matches)  
- ✅ Retrieval + LLM reasoning (for vague/natural queries like *"How much after expenses?"*)  

## ✨ Features
- Upload PDF/Excel financial statements  
- Extract tables and convert into structured rows  
- Embed data for semantic search  
- Ask financial questions in plain English  
- Get accurate answers via:
  - Direct lookup (e.g., "What is revenue?")
  - Reasoning with Ollama (e.g., "How much after expenses?")

## 🛠️ Tech Stack
- Python  
- Streamlit (UI)  
- pandas, pdfplumber (data extraction)  
- sentence-transformers (embeddings)  
- Ollama (local LLM - mistral / phi3 tested)  

## ⚙️ Setup Instructions

### 1️⃣ Clone Repo & Setup Environment
```bash
git clone https://github.com/your-username/financial-qa.git
cd financial-qa
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate   # On Mac/Linux
pip install -r requirements.txt
pip install -r requirements.txt
ollama pull mistral
ollama pull phi3
streamlit run app.py

---

### 5. Example Usage
```markdown
## 📂 Example Usage

Questions you can ask:
- `What is revenue?` → **1,200,000**  
- `Net income for 2024?` → **400,000**  
- `How much after expenses?` → **400,000 (Net Income)**  
## 👨‍💻 Author
**Anurag Mishra**  
(Data Science Internship Assignment Submission)
