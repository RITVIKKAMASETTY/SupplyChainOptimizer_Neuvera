import os
import streamlit.web.bootstrap

# Set the correct working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Run the Streamlit app
if __name__ == "__main__":
    streamlit.web.bootstrap.run(
        "DemandSupply/supply_chain_app.py",
        "streamlit run DemandSupply/supply_chain_app.py",
        [],
        {})
