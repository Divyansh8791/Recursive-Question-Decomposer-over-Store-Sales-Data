# 🧠 Recursive Data Assistant for Store Sales Analytics

A smart analytics assistant powered by **LangGraph** and **LangChain**, designed to answer both atomic and complex multi-hop questions over structured retail sales data.

---

## 📁 Dataset Overview

The system operates on a cleaned dataset with the following columns:

- `Store Name`  
- `Description`  
- `Department`  
- `Qty Sold`  
- `Cost`  
- `Retail`  
- `Total Retail`  
- `Margin`  
- `Profit`  

> **Note:** The dataset does **not contain any timestamp or date-based columns**. Time-based queries are gracefully handled by informing the user that such analysis is not possible due to the lack of temporal data.

---

## 💡 Features

- ✅ Handles **atomic** queries (e.g., "What is the total profit from HOT FOOD?")
- 🔁 Handles **complex** queries by:
  - Classifying complexity
  - Decomposing into sub-questions
  - Iteratively resolving each sub-question
  - Aggregating into a final structured JSON output
- 🧠 Uses a LangChain LLM to generate and execute valid Pandas code
- 📊 Executes safely using a controlled local environment
- 🖥️ Clean **Gradio UI** to interact with the assistant
- 📥 Option to **download final JSON output** of the reasoning process

---

## 🖼️ Sample Queries Handled

- Top 5 products by quantity sold in Golden LLC.
- Which product category (department) had the highest total revenue?
- What are the top-selling products by revenue in each store?

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Divyansh8791/Recursive-Question-Decomposer-over-Store-Sales-Data.git
cd Recursive-Question-Decomposer-over-Store-Sales-Data
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Add Environment Variables
```env
GOOGLE_API_KEY=your-google-api-key
```
### 4. Run the App
```bash
python app.py
```
---
## 📁 Project Structure
```bash
.
├── main.py        # Core logic, LangGraph setup and state handling
├── app.py         # Gradio UI frontend
├── cleaned_demo_sales_data.csv  # Input dataset
├── requirements.txt
└── README.md
```
---
## 📤 Submission Note
**This project was developed as a part of a take-home AI/ML internship assignment.
Time-based queries were handled gracefully with user communication since the dataset lacks timestamp features.**
---
## 🙋‍♂️ Author
**Divyansh Sharma**
