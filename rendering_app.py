import streamlit as st

st.set_page_config(page_title="State Modification from Python - Streamlit Flow", layout="wide")

st.title("State Modification from Python Demo")

# Removed st.echo('below') to prevent code output

import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import TreeLayout

if 'curr_state' not in st.session_state:
    nodes = [StreamlitFlowNode("1", (0, 0), {'content': 'Node 1'}, 'input', 'right'),
             StreamlitFlowNode("2", (1, 0), {'content': 'Node 2'}, 'default', 'right', 'left'),
             StreamlitFlowNode("3", (2, 0), {'content': 'Node 3'}, 'default', 'right', 'left'),
             ]

    edges = [StreamlitFlowEdge("1-2", "1", "2", animated=True),
             StreamlitFlowEdge("1-3", "1", "3", animated=True),
             ]

    st.session_state.curr_state = StreamlitFlowState(nodes, edges)

st.session_state.curr_state = streamlit_flow('example_flow',
                                             st.session_state.curr_state,
                                             layout=TreeLayout(direction='right'),
                                             fit_view=True,
                                             height=500,
                                             enable_node_menu=True,
                                             enable_edge_menu=True,
                                             enable_pane_menu=True,
                                             get_edge_on_click=True,
                                             get_node_on_click=True,
                                             show_minimap=False,
                                             hide_watermark=True,
                                             allow_new_edges=True,
                                             min_zoom=0.1)
