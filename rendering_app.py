import graphviz as graphviz
import streamlit as st
import uuid


@st.cache_data
def load_graph():
    # Create a graphlib graph object
    graph = graphviz.Digraph()
    graph.edge("run", "intr")
    graph.edge("intr", "runbl")
    graph.edge("runbl", "run")
    graph.edge("run", "kernel")
    graph.edge("kernel", "zombie")
    graph.edge("kernel", "sleep")
    graph.edge("kernel", "runmem")
    graph.edge("sleep", "swap")
    graph.edge("swap", "runswap")
    graph.edge("runswap", "new")
    graph.edge("runswap", "runmem")
    graph.edge("new", "runmem")
    graph.edge("sleep", "runmem")
    return graph


# Load and display the graph
graph = load_graph()
st.graphviz_chart(graph)


# Function to generate SVG data
def generate_svg_data(graph):
    # Render the graph as an SVG in memory
    svg_data = graph.pipe(format="svg").decode("utf-8")
    return svg_data


# Button to open the graph as SVG in a new tab
if st.button("Open SVG in New Tab", key="open_svg_button"):
    # Generate SVG data
    svg_data = generate_svg_data(graph)

    # Generate a unique identifier for each execution
    unique_id = str(uuid.uuid4())

    # JavaScript to open a new tab and write the SVG directly to the HTML
    js_code = f"""
    <script>
        var svgData = `{svg_data}`;  // Insert SVG content as a string
        var newTab = window.open("about:blank", "_blank");
        newTab.document.write('<html><head><title>SVG Image</title></head><body>' + svgData + '</body></html>');
        newTab.document.close();
    </script>
    """
    # Display the JavaScript in Streamlit to execute it
    st.components.v1.html(js_code + f"<!-- {unique_id} -->", height=0, width=0)
