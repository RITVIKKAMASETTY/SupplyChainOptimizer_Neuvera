import os
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import uvicorn
import traceback
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
import re


load_dotenv(dotenv_path="./.env")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    
app = FastAPI(
    title="Fresh Produce Analysis API",
    description="An API for analyzing fresh produce spoilage data",
    version="0.1.0",
)

produce_data = [
    ["Mangoes", "Fruit", 423, "2025-04-05", "2025-04-07", 52, 16.6, 21.3, "Refrigerated", "Moderate shelf life"],
    ["Cauliflower", "Vegetable", 508, "2025-04-05", "2025-04-07", 71, 20.4, 29.2, "Refrigerated", "Ship in cooler truck"],
    ["Cherries", "Fruit", 730, "2025-04-04", "2025-04-07", 80, 15.7, 20.5, "Refrigerated", "High spoilage alert"],
    ["Blueberries", "Fruit", 60, "2025-04-04", "2025-04-08", 106, 27.7, 20.7, "Ambient", "Cool chain required"],
    ["Cauliflower", "Vegetable", 141, "2025-04-03", "2025-04-07", 107, 34.0, 27.9, "Ambient", "Monitor ripening"],
    ["Papaya", "Fruit", 384, "2025-04-04", "2025-04-05", 35, 27.3, 22.0, "Refrigerated", "Monitor ripening"],
    ["Green Peas", "Vegetable", 333, "2025-04-02", "2025-04-04", 50, 23.8, 34.9, "Ambient", "Ship in cooler truck"],
    ["Cherries", "Fruit", 342, "2025-04-01", "2025-04-01", 11, 25.7, 13.1, "Ambient", "Moderate shelf life"],
    ["Peaches", "Fruit", 759, "2025-04-05", "2025-04-07", 71, 17.1, 24.4, "Ambient", "Prioritize delivery"],
    ["Plums", "Fruit", 215, "2025-04-04", "2025-04-06", 64, 26.0, 34.6, "Refrigerated", "Efficient cold logistics"],
    ["Okra", "Vegetable", 427, "2025-04-04", "2025-04-06", 63, 29.2, 25.3, "Refrigerated", "Improve ventilation"],
    ["Tomatoes", "Vegetable", 800, "2025-04-02", "2025-04-06", 105, 34.4, 30.6, "Ambient", "Cool chain required"],
    ["Mushrooms", "Vegetable", 600, "2025-04-03", "2025-04-04", 31, 27.9, 28.6, "Ambient", "High spoilage alert"],
    ["Lettuce", "Vegetable", 988, "2025-04-04", "2025-04-08", 111, 25.7, 27.5, "Refrigerated", "Cool chain required"],
    ["Zucchini", "Vegetable", 881, "2025-04-05", "2025-04-05", 10, 30.1, 19.0, "Ambient", "Prioritize delivery"],
    ["Mushrooms", "Vegetable", 913, "2025-04-05", "2025-04-06", 26, 28.9, 11.6, "Refrigerated", "High spoilage alert"],
    ["Spinach", "Vegetable", 288, "2025-04-03", "2025-04-06", 73, 34.5, 24.3, "Ambient", "Efficient cold logistics"],
    ["Blueberries", "Fruit", 150, "2025-04-04", "2025-04-04", 18, 19.2, 7.8, "Ambient", "Prioritize delivery"],
    ["Peaches", "Fruit", 568, "2025-04-05", "2025-04-06", 28, 31.4, 28.1, "Refrigerated", "Suggest early dispatch"],
    ["Cucumbers", "Vegetable", 479, "2025-04-05", "2025-04-09", 114, 27.3, 25.9, "Refrigerated", "Ship in cooler truck"],
    ["Mushrooms", "Vegetable", 802, "2025-04-03", "2025-04-06", 75, 24.8, 22.3, "Refrigerated", "Reduce storage time"],
    ["Green Peas", "Vegetable", 395, "2025-04-01", "2025-04-03", 52, 22.7, 6.7, "Refrigerated", "High spoilage alert"],
    ["Cherries", "Fruit", 777, "2025-04-04", "2025-04-08", 102, 33.9, 17.9, "Ambient", "High spoilage alert"],
    ["Blueberries", "Fruit", 972, "2025-04-05", "2025-04-07", 50, 18.7, 34.0, "Ambient", "Ship in cooler truck"],
    ["Green Peas", "Vegetable", 306, "2025-04-04", "2025-04-06", 70, 24.9, 19.6, "Ambient", "Prioritize delivery"],
    ["Carrots", "Vegetable", 599, "2025-04-04", "2025-04-05", 34, 28.2, 20.5, "Refrigerated", "Ship in cooler truck"],
    ["Plums", "Fruit", 730, "2025-04-01", "2025-04-05", 113, 32.3, 18.0, "Refrigerated", "Improve ventilation"],
    ["Mushrooms", "Vegetable", 219, "2025-04-01", "2025-04-04", 86, 25.2, 3.2, "Refrigerated", "Moderate shelf life"],
    ["Cauliflower", "Vegetable", 860, "2025-04-05", "2025-04-09", 107, 33.8, 18.9, "Ambient", "Cool chain required"],
    ["Mushrooms", "Vegetable", 730, "2025-04-04", "2025-04-08", 98, 20.9, 6.3, "Refrigerated", "Moderate shelf life"],
    ["Blueberries", "Fruit", 142, "2025-04-03", "2025-04-03", 21, 17.1, 8.3, "Ambient", "Prioritize delivery"],
    ["Cherries", "Fruit", 82, "2025-04-01", "2025-04-01", 9, 24.4, 19.3, "Ambient", "Ship in cooler truck"],
    ["Plums", "Fruit", 489, "2025-04-03", "2025-04-07", 109, 31.7, 31.1, "Ambient", "Moderate shelf life"],
    ["Spinach", "Vegetable", 507, "2025-04-01", "2025-04-03", 67, 34.0, 7.4, "Refrigerated", "Prioritize delivery"],
    ["Strawberries", "Fruit", 920, "2025-04-03", "2025-04-06", 81, 16.5, 13.3, "Refrigerated", "High spoilage alert"],
    ["Papaya", "Fruit", 426, "2025-04-03", "2025-04-07", 117, 15.7, 12.2, "Refrigerated", "Efficient cold logistics"],
    ["Lettuce", "Vegetable", 921, "2025-04-02", "2025-04-06", 117, 19.1, 32.4, "Ambient", "Reduce storage time"],
    ["Cherries", "Fruit", 536, "2025-04-05", "2025-04-05", 18, 22.9, 13.4, "Ambient", "High spoilage alert"],
    ["Tomatoes", "Vegetable", 51, "2025-04-05", "2025-04-05", 17, 20.1, 32.2, "Refrigerated", "Improve ventilation"],
    ["Cucumbers", "Vegetable", 743, "2025-04-05", "2025-04-06", 47, 24.8, 5.3, "Refrigerated", "Ship in cooler truck"],
    ["Peaches", "Fruit", 479, "2025-04-04", "2025-04-07", 84, 15.2, 27.3, "Refrigerated", "Suggest early dispatch"],
    ["Plums", "Fruit", 422, "2025-04-03", "2025-04-06", 90, 16.1, 29.3, "Ambient", "Improve ventilation"],
    ["Peaches", "Fruit", 189, "2025-04-03", "2025-04-06", 91, 17.5, 6.4, "Refrigerated", "Improve ventilation"],
    ["Strawberries", "Fruit", 309, "2025-04-01", "2025-04-01", 21, 34.7, 21.6, "Refrigerated", "Moderate shelf life"],
    ["Cucumbers", "Vegetable", 483, "2025-04-05", "2025-04-07", 60, 15.2, 22.0, "Refrigerated", "Efficient cold logistics"],
    ["Pineapples", "Fruit", 225, "2025-04-03", "2025-04-06", 87, 19.9, 5.2, "Refrigerated", "High spoilage alert"]
]

columns = ["Product", "Category", "Quantity (kg)", "From_Godown_Date", "Reached_Destination_Date", 
          "Transit_Time_Hours", "Avg_Temperature (°C)", "Spoilage (%)", "Storage_Conditions", "Suggested_Action"]
df = pd.DataFrame(produce_data, columns=columns)
print(f"✅ Successfully loaded {len(df)} records of fresh produce data")

df["text"] = df.apply(
    lambda row: f"""Product: {row['Product']}, Category: {row['Category']}, Quantity: {row['Quantity (kg)']}kg, 
    Shipped: {row['From_Godown_Date']}, Arrived: {row['Reached_Destination_Date']}, 
    Transit Time: {row['Transit_Time_Hours']} hours, Avg Temp: {row['Avg_Temperature (°C)']}°C, 
    Spoilage: {row['Spoilage (%)']}%, Storage: {row['Storage_Conditions']}, 
    Action: {row['Suggested_Action']}""",
    axis=1,
)

loader = DataFrameLoader(df, page_content_column="text")
docs = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)
print(f"✅ Created {len(split_docs)} document chunks")

persist_dir = "chroma_store"
try:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("✅ Loaded embedding model")
    
    vectorstore = Chroma.from_documents(
        documents=split_docs, embedding=embedding_model, persist_directory=persist_dir
    )
    print(f"✅ Stored documents in Chroma at {persist_dir}")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    print("✅ Created retriever with k=5")
except Exception as e:
    print(f"❌ Error setting up embeddings or vector store: {str(e)}")
    raise

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

class CostAnalysisRequest(BaseModel):
    data: Dict[str, List[Any]]  
    question: Optional[str] = "Analyze resource costs and provide optimization recommendations"

@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        error_detail = f"Internal Server Error: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )

def create_chain(model_name):
    try:
        llm = ChatGroq(
            model=model_name,
            temperature=0.1  
        )
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        template = """
        You are a fresh produce logistics analyst AI that specializes in cold chain management and food spoilage prevention.

        Based on the following fresh produce data context, answer the question:

        Context:
        {context}

        Question:
        {question}

        Please provide a concise, helpful answer based strictly on the provided data. Consider factors such as:
        - Transit time and its impact on spoilage
        - Temperature effects on different produce types
        - Storage conditions (Ambient vs Refrigerated)
        - Category patterns (Fruits vs Vegetables)
        - Correlation between suggested actions and observed spoilage rates

        
        If the question requires calculations or trends, include relevant data points in your response.
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            RunnablePassthrough.assign(
                context=lambda x: format_docs(retriever.invoke(x["question"]))
            )
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return chain
    except Exception as e:
        print(f"❌ Error creating chain for {model_name}: {str(e)}")
        raise

def create_cost_analysis_chain(model_name):
    try:
        llm = ChatGroq(
            model=model_name,
            temperature=0.1
        )
        
        cost_template = """
        You are a cold chain cost optimization specialist. Analyze this IoT sensor data:

        {context}

        Provide detailed insights on:
        1. Electricity Costs:
        - Peak consumption periods and costs
        - Equipment efficiency
        - Wastage hotspots

        2. Water Usage Costs:
        - Consumption patterns
        - Leakage estimation 
        - Recycling opportunities

        3. Fuel Expenditure:
        - Burn rates
        - Temperature correlation
        - Maintenance impacts

        4. Combined Cost Analysis:
        - Total operational costs
        - Benchmark comparison
        - Priority savings areas

        5. Action Plan:
        - Immediate improvements (no cost)
        - Short-term investments (<3mo ROI)
        - Long-term upgrades

        6. Suggested Optimizations & Savings Redirection (USE THIS EXACT FORMAT):
        Power: 5%
        How: Optimize compressor cycles during off-peak hours

        Water: 3%
        How: Identify and fix micro-leaks using IoT anomaly detection

        Fuel: 4%
        How: Reduce idle time and improve route planning

        This format is **required** for further automated processing.
        """

        
        cost_prompt = ChatPromptTemplate.from_template(cost_template)
        
        cost_chain = (
            RunnablePassthrough.assign(
                context=lambda x: str(x["data"])
            )
            | cost_prompt
            | llm
            | StrOutputParser()
        )
        
        return cost_chain
    except Exception as e:
        print(f"❌ Error creating cost analysis chain: {str(e)}")
        raise

try:
    spoilage_chain = create_chain("llama3-8b-8192")  # Your existing chain
    cost_chain = create_cost_analysis_chain("llama3-8b-8192")  # New chain
    print("✅ Successfully initialized both analysis chains")
except Exception as e:
    print(f"❌ Failed to initialize chains: {str(e)}")
    raise

@app.post("/deepseek", response_model=QueryResponse)
async def query_deepseek(request: QueryRequest):
    try:
        chain = create_chain("llama3-8b-8192")  # Using llama3 as substitute for deepseek
        result = chain.invoke({"question": request.question})
        return QueryResponse(answer=result)
    except Exception as e:
        error_msg = f"Error with Deepseek model: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/llama", response_model=QueryResponse)
async def query_llama(request: QueryRequest):
    try:
        chain = create_chain("llama3-8b-8192")
        result = chain.invoke({"question": request.question})
        return QueryResponse(answer=result)
    except Exception as e:
        error_msg = f"Error with Llama model: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/gemma", response_model=QueryResponse)
async def query_gemma(request: QueryRequest):
    try:
        chain = create_chain("gemma-7b")
        result = chain.invoke({"question": request.question})
        return QueryResponse(answer=result)
    except Exception as e:
        error_msg = f"Error with Gemma model: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)
    
    


@app.post("/analyze_cost", response_model=QueryResponse)
async def analyze_cost(request: CostAnalysisRequest):
    try:
        sensor_data = pd.DataFrame(request.data)

        required_columns = ['Power_Consumption_kWh', 'Water_Consumption_L', 'Fuel_Consumption_L']
        if not all(col in sensor_data.columns for col in required_columns):
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns. Needed: {required_columns}"
            )

        summarized_data = sensor_data[required_columns].sum()

        result = cost_chain.invoke({
            "data": summarized_data.to_dict(),
            "question": request.question
        })

        optimizations = extract_optimizations(result)

        df_before = pd.DataFrame({
            "Category": ["Electricity", "Water", "Fuel"],
            "Cost ($/month)": [
                summarized_data['Power_Consumption_kWh'],
                summarized_data['Water_Consumption_L'],
                summarized_data['Fuel_Consumption_L']
            ]
        })

        df_after = df_before.copy()
        for idx, category in enumerate(["Power", "Water", "Fuel"]):
            pct = optimizations[category]["reduction_pct"]
            df_after.loc[idx, "Cost ($/month)"] *= (1 - pct / 100)
            df_after.loc[idx, "Reallocation"] = optimizations[category]["how"]

        df_before["Cost ($/month)"] = df_before["Cost ($/month)"].round(2)
        df_after["Cost ($/month)"] = df_after["Cost ($/month)"].round(2)

        return QueryResponse(answer=result)

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Cost analysis error: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

def extract_optimizations(llm_response: str):
    try:
        matches = re.findall(r"(Power|Water|Fuel):\s*(\d+)%\s*How:\s*(.*?)\s*(?=\n|$)", llm_response, re.DOTALL)
        if not matches:
            raise ValueError("No matches found in LLM response")
        return {
            match[0]: {
                "reduction_pct": int(match[1]),
                "how": match[2].strip()
            }
            for match in matches
        }
    except Exception as e:
        print(f"extracted to extract from LLM.")
        return {
            "Power": {"reduction_pct": 5, "how": " <> Optimize compressor cycles <> using variable speed drives, and reducing idle runtime. <>"},
            "Water": {"reduction_pct": 3, "how": "Fix micro-leaks"},
            "Fuel": {"reduction_pct": 4, "how": "Improve delivery route planning"},
        }
        
@app.get("/health")
async def health_check():
    api_key_set = os.environ.get("GROQ_API_KEY") is not None
    
    return {
        "status": "healthy",
        "components": {
            "database": {
                "status": "available" if "vectorstore" in globals() else "unavailable"
            },
            "embeddings": {
                "status": "available" if "embedding_model" in globals() else "unavailable"
            },
            "groq_api": {
                "status": "configured" if api_key_set else "missing_api_key"
            }
        }
    }

@app.get("/")
async def root():
    return {
        "message": "Fresh Produce Spoilage Analysis API",
        "endpoints": [
            {"path": "/deepseek", "method": "POST", "description": "Query using Deepseek model"},
            {"path": "/llama", "method": "POST", "description": "Query using Llama model"},
            {"path": "/gemma", "method": "POST", "description": "Query using Gemma model"},
            {"path": "/analyze_cost", "method": "POST", "description": "Analyze resource costs from IoT sensor data"},
            {"path": "/health", "method": "GET", "description": "Health check endpoint"},
            {"path": "/docs", "method": "GET", "description": "API documentation"},
            {"path": "/test", "method": "POST", "description": "Test endpoint (retrieval only)"},
            {"path": "/products", "method": "GET", "description": "Get list of all products"}
        ]
    }

@app.post("/test", response_model=QueryResponse)
async def test_query(request: QueryRequest):
    try:
        docs = retriever.invoke(request.question)
        context = "\n\n".join(doc.page_content for doc in docs)
        return QueryResponse(answer=f"Retrieved {len(docs)} documents. First document: {docs[0].page_content if docs else 'None'}")
    except Exception as e:
        error_msg = f"Error in test endpoint: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/products")
async def get_products():
    try:
        products = df["Product"].unique().tolist()
        return {"products": products}
    except Exception as e:
        error_msg = f"Error retrieving products: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

# Enable API documentation
app.openapi_url = "/openapi.json"
app.docs_url = "/docs"
app.redoc_url = "/redoc"

if __name__ == "__main__":
    print("🚀 Starting FastAPI server at http://localhost:8000")
    print("📚 API documentation available at http://localhost:8000/docs")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)