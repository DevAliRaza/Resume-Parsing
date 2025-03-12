import matplotlib.pyplot as plt
from pypdf import PdfReader
import re
import numpy as np
from numpy.linalg import norm
import streamlit as st
from sentence_transformers import SentenceTransformer
import os
from pinecone import Pinecone

st.markdown(
    """
    <h1 style="text-align: center; color: #4CAF50; font-size: 48px;">
        RESUME SCREENING USING NLP
    </h1>
    """,
    unsafe_allow_html=True,
)
st.write("Upload a PDF file, and its content will be displayed below.")

def c(text):
    a = text.count("c/c++")
    return len(re.findall(r"(\t|\n|,|\.| |^)c( |$|\t|\n|,|\.)", text)) + a

def cpp(text):
    a = len(re.findall(r"(\t|\n|,|\.| |^)cpp( |$|\t|\n|,|\.)", text))
    b = len(re.findall(r"(\t|\n|,|\.| |^)c\+\+( |$|\t|\n|,|\.)", text))
    c = text.count("c/c++")
    return a + b + c

def assembly(text):
    return text.count("assembly")

def java(text):
    return len(re.findall(r"(\t|,|\.|\n| |^)java( |$|\t|,|\.|\n)", text))

def js(text):
    return len(re.findall(r"(\t|,|\.|\n| |^)js( |$|\t|,|\.|\n)", text))

def objC(text):
    return text.count("objective c") + text.count("objective-c")

def datastructs(text):
    return sum(text.count(term) for term in ["datastructures", "data-structures", "data structures"])

def socket(text):
    return len(re.findall(r"(\t|,|\.|\n| |^)socket( |$|\t|\n|,|\.)", text))

def sql(text):
    return len(re.findall(r"(\t|\n|,|\.| |^)sql( |$|\t|,|\.|\n)", text))

def go(text):
    return len(re.findall(r"(\t|\n|,|\.| |^)go( |$|\t|\n|,|\.)", text))

def opencv(text):
    return text.count("opencv")

def boost(text):
    return text.count("boost")

def eigen(text):
    return text.count("eigen")

def numpy_func(text):
    return text.count("numpy")

def scikit_learn(text):
    return len(re.findall(r"scikit-learn", text, re.IGNORECASE))

def keras(text):
    return text.count("keras")

def matplotlib_func(text):
    return text.count("matplotlib")

def seaborn(text):
    return text.count("seaborn")

def nodejs(text):
    return len(re.findall(r"(node\.js|nodejs)", text, re.IGNORECASE))

def expressjs(text):
    return len(re.findall(r"(express\.js|expressjs)", text, re.IGNORECASE))

def jquery(text):
    return len(re.findall(r"(jquery|jQuery)", text))

def unity(text):
    return text.count("unity")

def unreal_engine(text):
    return len(re.findall(r"(unreal engine|unrealengine)", text, re.IGNORECASE))

def pygame(text):
    return text.count("pygame")

dims = {
    "c": 0,
    "cpp": 1,
    "assembly": 2,
    "python": 3,
    "java": 4,
    "javascript": 5,
    "html": 6,
    "css": 7,
    "lua": 8,
    "ruby": 9,
    "rust": 10,
    "go": 11,
    "swift": 12,
    "objective-c": 13,
    "android development": 14,
    "ios development": 15,
    "mobile development": 16,
    "web development": 17,
    "machine learning": 18,
    "data science": 19,
    "embedded development": 20,
    "data structures": 21,
    "tensorflow": 22,
    "pytorch": 23,
    "pandas": 24,
    "reactjs": 25,
    "nextjs": 26,
    "vue": 27,
    "angular": 28,
    "bootstrap": 29,
    "sdl": 30,
    "flex": 31,
    "bison": 32,
    "llvm": 33,
    "compiler": 34,
    "libcurl": 35,
    "websockets": 36,
    "socket": 37,
    "springboot": 38,
    "laravel": 39,
    "django": 40,
    "fastapi": 41,
    "flask": 42,
    "sql": 43,
    "mongodb": 44,
    "mysql": 45,
    "postgresql": 46,
    "mssql": 47,
    "qt": 48,
    "php": 49,
    "opencv": 50,
    "boost": 51,
    "eigen": 52,
    "numpy": 53,
    "scikit-learn": 54,
    "keras": 55,
    "matplotlib": 56,
    "seaborn": 57,
    "node.js": 58,
    "express.js": 59,
    "jquery": 60,
    "unity": 61,
    "unreal engine": 62,
    "pygame": 63,
    "gtk": 64,
    "glib": 65,
    "openssl": 66,
    "ncurses": 67,
    "tailwindcss": 68,
    "bulma": 69,
    "foundation": 70,
    "svelte": 71,
    "nuxtjs": 72,
    "xgboost": 73,
    "lightgbm": 74,
    "flutter": 75,
    "react native": 76,
    "kotlin": 77,
    "gcc": 78,
    "clang": 79,
    "yacc": 80,
    "hibernate": 81,
    "junit": 82,
    "maven": 83,
    "gradle": 84,
    "kafka": 85,
    "sqlite": 86,
    "oracle db": 87,
    "aws": 88,
    "azure": 89,
    "gcp": 90,
    "docker": 91,
    "kubernetes": 92,
    "terraform": 93,
    "sass": 94,
    "less": 95,
    "parcel": 96,
    "vite": 97,
    "frontend": 98,
    "backend": 99,
    "general programming": 100,
    "ruby on rails": 101,
    "scipy": 102,
    "beautifulsoup": 103,
    "scrapy": 104,
    "three.js": 105,  
    "d3.js": 106,
    "cloud":107,
}

def generate_embedding(text):
    return [
        c(text),
        cpp(text),
        assembly(text),
        text.count("python"),
        java(text),
        text.count("javascript") + js(text),
        text.count("html"),
        text.count("css"),
        text.count("lua"),
        text.count("ruby"),
        text.count("rust"),
        go(text),
        text.count("swift"),
        objC(text),
        text.count("android"),
        text.count("ios"),
        text.count("mobile development"),
        text.count("web"),
        text.count("machine learning"),
        text.count("data science"),
        text.count("embedded development"),
        datastructs(text),
        text.count("tensorflow"),
        text.count("pytorch"),
        text.count("pandas"),
        text.count("reactjs"),
        text.count("nextjs"),
        text.count("vue"),
        text.count("angular"),
        text.count("bootstrap"),
        text.count("sdl"),
        text.count("flex"),
        text.count("bison"),
        text.count("llvm"),
        text.count("compiler"),
        text.count("libcurl"),
        text.count("websocket"),
        socket(text),
        text.count("springboot"),
        text.count("laravel"),
        text.count("django"),
        text.count("fastapi"),
        text.count("flask"),
        sql(text),
        text.count("mongodb"),
        text.count("mysql"),
        text.count("postgresql"),
        text.count("mssql"),
        text.count("qt"),
        text.count("php"),
        opencv(text),
        boost(text),
        eigen(text),
        numpy_func(text),
        scikit_learn(text),
        keras(text),
        matplotlib_func(text),
        seaborn(text),
        nodejs(text),
        expressjs(text),
        jquery(text),
        unity(text),
        unreal_engine(text),
        pygame(text),
        text.count("gtk"),
        text.count("glib"),
        text.count("openssl"),
        text.count("ncurses"),
        text.count("tailwindcss"),
        text.count("bulma"),
        text.count("foundation"),
        text.count("svelte"),
        text.count("nuxtjs"),
        text.count("xgboost"),
        text.count("lightgbm"),
        text.count("flutter"),
        text.count("react native"),
        text.count("kotlin"),
        text.count("gcc"),
        text.count("clang"),
        text.count("yacc"),
        text.count("hibernate"),
        text.count("junit"),
        text.count("maven"),
        text.count("gradle"),
        text.count("kafka"),
        text.count("sqlite"),
        text.count("oracle db"),
        text.count("aws"),
        text.count("azure"),
        text.count("gcp"),
        text.count("docker"),
        text.count("kubernetes"),
        text.count("terraform"),
        text.count("sass"),
        text.count("less"),
        text.count("parcel"),
        text.count("vite"),
        text.count("frontend"),
        text.count("backend"),
        text.count("general programming"),
        text.count("ruby on rails"),
        text.count("beautifulsoup"),
        text.count("scipy"),
        text.count("scrapy"),
        text.count("three.js"),
        text.count("d3.js"),
        text.count("cloud")
    ]

def add_weighted_sum(vec, to_update, keys, weights):
    i = 0
    for key in keys:
        vec[dims[to_update]] += vec[dims[key]] * weights[i]
        i += 1

def add_dep_score(vec):
    # SDL, Flex, Bison are C libraries
    add_weighted_sum(vec, "c", ["sdl", "flex", "bison", "gtk", "glib", "libcurl", "openssl", "ncurses"], [0.1] * 8)
    # C influences C++
    vec[dims["cpp"]] += 0.4 * vec[dims["c"]]
    # TensorFlow, PyTorch, and other ML libraries improve machine learning skills
    add_weighted_sum(vec, "machine learning", ["pytorch", "tensorflow", "scikit-learn", "keras", "xgboost", "lightgbm"], [0.5] * 6)
    # Bootstrap is CSS
    add_weighted_sum(vec, "css", ["bootstrap", "tailwindcss", "bulma", "foundation"], [0.1] * 4)
    add_weighted_sum(
        vec, 
        "web development", 
        [
            "django", "fastapi", "laravel", "flask", "reactjs", "vue", "angular", "svelte", 
            "nextjs", "nuxtjs", "express.js", "ruby on rails", "html", "css", "javascript"
        ], 
        [0.2] * 15
    )
    add_weighted_sum(
        vec, 
        "python", 
        ["django", "flask", "pytorch", "tensorflow", "pandas", "numpy", "scipy", "matplotlib", "seaborn", "fastapi"], 
        [0.1] * 10
    )
    add_weighted_sum(
        vec, 
        "javascript", 
        ["reactjs", "vue", "angular", "nextjs", "nuxtjs", "jquery", "d3.js", "three.js", "svelte"], 
        [0.1] * 9
    )
    add_weighted_sum(vec, "mobile development", ["android development", "ios development", "flutter", "react native", "kotlin"], [0.5] * 5)
    # Flex, Bison, LLVM improve compiler construction skills
    add_weighted_sum(vec, "compiler", ["llvm", "flex", "bison", "gcc", "clang", "yacc"], [0.7] * 6)
    # Qt improves C++ skills
    add_weighted_sum(vec, "cpp", ["qt", "boost", "opencv", "eigen"], [0.3] * 4)
    # Java frameworks and libraries
    add_weighted_sum(vec, "java", ["springboot", "hibernate", "junit", "maven", "gradle", "kafka"], [0.1] * 6)
    # SQL tools and databases
    add_weighted_sum(vec, "sql", ["mysql", "mssql", "postgresql", "sqlite", "mongodb", "oracle db"], [0.2] * 6)
    # Cloud tools and DevOps influence skills in these areas
    add_weighted_sum(vec, "cloud", ["aws", "azure", "gcp", "docker", "kubernetes", "terraform"], [0.3] * 6)
    # Frontend development includes additional CSS and JS libraries
    add_weighted_sum(vec, "frontend", ["tailwindcss", "sass", "less", "bootstrap", "foundation", "parcel", "vite"], [0.2] * 7)
    # Backend development includes Node.js, Express, and others
    add_weighted_sum(vec, "backend", ["express.js", "fastapi", "flask", "django", "laravel", "springboot"], [0.2] * 6)
    # Additional general-purpose libraries for programming
    add_weighted_sum(vec, "general programming", ["numpy", "pandas", "matplotlib", "scipy", "beautifulsoup", "scrapy"], [0.2] * 6)

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Read the uploaded PDF
    reader = PdfReader(uploaded_file)
    text = ''
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + '\n'
    
    text = text.lower()
    
    # Load the pre-trained all-MiniLM-L6-v2 model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Tokenize the text into chunks (512 tokens max per chunk)
    def split_text_into_chunks(text, max_tokens=512):
        """
        Split the text into chunks of max_tokens, handling tokenization limits.
        """
        tokenizer = model.tokenizer
        tokens = tokenizer.encode(text)  # Tokenize text
        chunks = []
        
        for i in range(0, len(tokens), max_tokens):
            chunks.append(tokens[i:i + max_tokens])
        
        return chunks
    
    # Split the CV text into chunks
    chunks = split_text_into_chunks(text, max_tokens=512)
    
    # Generate embeddings for each chunk
    transformer_cv_embeddings_arr = []
    for chunk in chunks:
        # Decode chunk tokens back to text (for batch processing) and generate embedding
        chunk_text = model.tokenizer.decode(chunk)
        chunk_embedding = model.encode(chunk_text)
        transformer_cv_embeddings_arr.append(chunk_embedding)
    
    # Convert transformer_cv_embeddings_arr to a numpy array
    transformer_cv_embeddings_arr = np.array(transformer_cv_embeddings_arr)
    
    if transformer_cv_embeddings_arr.size == 0:
        transformer_cv_embedding = np.zeros(model.get_sentence_embedding_dimension())
    else:
        # Average the embeddings (to get a single representation of the entire CV)
        transformer_cv_embedding = transformer_cv_embeddings_arr.mean(axis=0)
    
    cv_vec = generate_embedding(text)
    ivd = {v: k for k, v in dims.items()}
    st.markdown(
                """
                     <h3 style="text-align: center; color: #4CAF50; font-size: 48px;">
                   BELOW ARE THE SKILLS OF THIS CANDIDATE
                </h3>
            """,
    unsafe_allow_html=True,
)
    for i in range(len(cv_vec)):
        if cv_vec[i] != 0 and i in ivd:

            st.write(ivd[i])
            st.write("-" * 50)

    
    add_dep_score(cv_vec)
    
    weight = 0.5
    cv_embedding = np.concatenate(
        [weight * np.array(cv_vec), (1 - weight) * np.array(transformer_cv_embedding)]
    )
    
    # Set the API key
    os.environ["PINECONE_API_KEY"] = "pcsk_2BVuyU_8HtemHZ7ZA2CQGFpbhUePfNfcu4pZA9RgKrGvYPYge7qWA9v9QsUFMGt9UW3xLs"  # Pinecone API key
    
    # Initialize Pinecone client
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    
    # Connect to Pinecone index
    index_name = "job-embedding"  # Replace with your Pinecone index name
    index = pc.Index(index_name)
    
    # Function to query Pinecone for similar items based on the given embedding
    def query_pinecone(precomputed_embedding, index, top_k=5, similarity_threshold=0.5):
        try:
            query_embedding_list = precomputed_embedding.tolist()  # Ensure it's a list
            query_results = index.query(
                vector=query_embedding_list,  # The query vector as a list
                top_k=top_k,                  # Number of closest matches to return
                include_metadata=True         # Include metadata in the results
            )
    
            matches = []
            for match in query_results['matches']:
                score = match.get('score', 0)  # Get similarity score
                if score >= similarity_threshold:
                    matches.append(match)
            
            return matches
        except Exception as e:
            st.error(f"Error during Pinecone query: {e}")
            return []
    
    # Query Pinecone for results
    top_k_results = query_pinecone(cv_embedding, index, top_k=10)
    
    # Format and display the results
    st.write("### Below are the relevant Job positions and their similarity indexes for the candidate")
    
    if top_k_results:
        for i, match in enumerate(top_k_results, start=1):
            metadata = match.get('metadata', {})
            id_ = match.get('id', 'N/A')
            score = match.get('score', 0)
    
            # Display results
            st.write(f"**{i}. Result {i}:**")
            st.write(f"■ **ID:** {id_}")
            st.write(f"■ **Metadata:** {metadata.get('job', 'N/A')}")
            st.write(f"■ **Similarity Score:** {score:.2f}")
            st.write("-" * 50)
    else:
        st.write("No results found above the similarity threshold.")
    


