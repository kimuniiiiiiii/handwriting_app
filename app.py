import streamlit as st
import src.handwriting

st.set_page_config(
		page_title="Data Science apps by Yuya Kimura",
		page_icon="🌏",
		layout="wide",
		initial_sidebar_state="expanded",
	)

PAGES = {
	# "Home": src.home, 
	# "About": src.about, 
	# "Descriptive statistics": src.statistics, 
	# "Map the data": src.map, 
	# "Explore weather conditions": src.weather, 
	"Hand Writing digits recognition": src.handwriting, 
	# "Model mode choices": src.model, 
}

def main():	
	# application architecture
    with st.sidebar:
        st.title("Data Science apps by Yuya Kimura 🌏")
        st.subheader("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    
    page = PAGES[selection]
    with st.spinner(f"Loading {selection} ..."):
        page.write()
    
    st.sidebar.info("**Author, Developer:** [Yuya Kimura](https://github.com/kimuniiiiiiii)")

if __name__ == "__main__":
	main()