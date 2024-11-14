import graphviz
import streamlit as st
import uuid

# Define the Graphviz script as a Python string
graph_script = """
dot = graphviz.Digraph()
dot.node("A", "Receive Invoice")
dot.node("B", "Verify Invoice Legitimacy")
dot.node("C", "Is Invoice Legitimate?", shape="diamond")
dot.node("D", "Contact Vendor for Verification")
dot.node("E", "Flag for Further Review & Involve Supervisor")
dot.edge("A", "B")
dot.edge("B", "C")
dot.edge("C", "D", label="No")
dot.edge("C", "E", label="Yes")
"""

# Function to dynamically execute the graph script and return the graph object
def load_graph_from_script(script):
    # Execute the script to create the `dot` object
    local_vars = {"graphviz": graphviz}  # Pass graphviz into the exec context
    exec(script, {}, local_vars)
    dot = local_vars["dot"]  # Retrieve the `dot` object from local variables
    return dot

# Load and display the graph using the script in the variable
dot = load_graph_from_script(graph_script)
st.graphviz_chart(dot)

# Function to generate SVG data from `dot`
def generate_svg_data(dot):
    # Render the graph as an SVG in memory
    svg_data = dot.pipe(format="svg").decode("utf-8")
    return svg_data

# Button to open the graph as SVG in a new tab
if st.button("Open SVG in New Tab", key="open_svg_button"):
    # Generate SVG data from the graph
    svg_data = generate_svg_data(dot)

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
