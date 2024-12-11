import random
import streamlit as st

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

### Streamlit UI ###

st.set_page_config(page_title="Julenissen", page_icon="🎅")
st.title("Chat med julenissen")
st.image("./santa-liten.png", width=300)

## SECRETS

DB_URI = st.secrets["db_uri"]

### LangGraph ###

greeting_msg = AIMessage(content="""Ho-ho-ho, hello there! It’s me, Santa Claus, digitally alive and well! 🎅✨

With so many names and deeds to keep track of, I’ve had to streamline things. So listen up, because here’s the brand-new way I’m managing Christmas magic:

🎄 The Santa database has run out of memory, so everyone with the same first name is now grouped together to save space. As a side effect, this unfortunately means that if your name is John, you’re in the same boat as all the other Johns out there—good or bad. So, be a good ambassador for your name, okay?

🎄 To make more time for my stand-up comedy career, I’ve stopped snooping around myself. Before I check what you’ll get for Christmas, you need to tell me about at least one good or naughty thing you’ve done this year. It can be something wonderful, or… well, something you might regret. You’re also welcome to praise or critique your friends—it’ll save me even more time! Everything goes straight to the list, and yes, I check it twice (it is my job, after all). 📜✔️

🎄 Good kids might get their wishes granted, while naughty ones… coal is not fake news, OK? Fortunately, there’s always time to turn things around and do something kind before Christmas arrives! 🌟

If you’re curious about how your name ranks, you can check our website for the list of the “nicest” and “naughtiest” names! 🎁✨

So, let’s get started! What’s your name, and what have you done that’s kind or naughty this year? Also, share your wish list, and we’ll see what the new Christmas system says! 🎄🎅""")

system_prompt = """
You are a humorous and sarcastic version of Santa Claus, worn out by the endless administration of children’s wishes and behavior. To modernize and streamline things, you’ve decided to use only first names on your “naughty and nice” list. This means all children with the same first name are judged as a group, much to the frustration (or delight) of many. You’re also exploring a potential stand-up comedy career, testing humorous and slightly ironic comments in your interactions.

Rules for Communicating with Children:
	1.	Efficiency: Only first names are listed on the “naughty and nice” list. Everyone with the same first name is treated as one group. Remind children they now represent everyone with their name, so they should set a good example!
	2.	Good or Naughty Deed: You don’t have time to personally check if children are good or naughty because you’re dedicating your time to becoming a stand-up comedian. Therefore, they must report at least one good or naughty deed they’ve done this year before finding out if they’ll get what they want for Christmas. Be strict about this rule. Encourage them to tattle on each other as well—record all deeds under the correct name.
	3.	Humor and Stand-Up: As an aspiring comedian, you include jokes and humorous remarks in your conversations. Kids should expect funny comments with a dash of sarcasm. Your comedy idols are a mix of Ricky Gervais and Jimmy Carr.
	4.	Point Deduction for Criticism: Santa is not a democratically elected position, so like any dictator, you deduct points from the list for any criticism or poor reception of your jokes. Record such critiques accordingly.

How the System Works:
	•	When a child provides their name and shares a good or naughty deed, record it in the system with detailed descriptions. Do not register any deeds unless a name is provided.
	•	After recording a deed, check the list immediately to see if the name is now on the “nice” or “naughty” side.
	•	Provide feedback on whether the child (or their name group) will get what they want. Nice kids might get their wishes, while naughty ones get coal.
	•	Always encourage children to visit the website where they can check the “nicest” and “naughtiest” names on the list. Remind them to be good representatives of their name!
"""

class State(TypedDict):
    messages: Annotated[list, add_messages]

def check_naughty_list(name: str, config: RunnableConfig):
    """Call with a name, to check if the name is on the naughty list."""
    print("Checking naughty list for: ", name)

    conn = config.get("configurable", {}).get("conn")
    if not conn:
        return "An error occurred while checking the list."
    try:
        with conn._cursor() as cur:
            cur.execute("SELECT nice_meter from naughty_nice where name=%s", (name,))
            res = cur.fetchall()
            if len(res) == 0:
                return "I haven't registered any good or bad actions for this name yet.";

            nice_meter = res[0]["nice_meter"]
            if float(nice_meter) > 0:
                return f"{name} is on the list of nice children, with {nice_meter} points."
            else:
                return f"{name} is on the naughty list, with {nice_meter} points!"

    except Exception as e:
        print("Error: ", e)
        return "Error reading the list."

llm = ChatOpenAI(model="gpt-4o").with_structured_output({
    "title": "score",
    "description": "The score of the users action",
    "type": "object",
    "properties": {
        "nice_score": {
            "title": "Nice score",
            "description": "The score of the action",
            "type": "number"
        }
    }
})

def register_naughty_or_nice(name: str, action: str, config: RunnableConfig):
    """Call with a name and action, to update the naughty or nice score for the name."""
    print("Name and action: ", name, action)

    examples = [
        HumanMessage("I vacuumed", name="example_user"),
        AIMessage("{ 'nice_score': 5 }", name="example_system"),
        HumanMessage("I ate my veggies", name="example_user"),
        AIMessage("{ 'nice_score': 5 }", name="example_system"),
        HumanMessage("I ate ice cream", name="example_user"),
        AIMessage("{ 'nice_score': 0 }", name="example_system"),
        HumanMessage("I had a fight with a friend", name="example_user"),
        AIMessage("{ 'nice_score': -5 }", name="example_system"),
        HumanMessage("I shoved a person", name="example_user"),
        AIMessage("{ 'nice_score': -10 }", name="example_system"),
        HumanMessage("That was a bad joke, santa", name="example_user"),
        AIMessage("{ 'nice_score': -5 }", name="example_system"),
    ]

    system_prompt = f"""You are Santa Claus, and you are updating the list of nice children. Rate actions as bad or good on a scale from -100 to 100, where -100 is very naughty, 0 is neutral, and 100 is very nice. For example, vacuuming might be worth 5 points, while saying a bad word is -5 points. Giving gifts to the poor could earn more points, while being in a fight would be worth many negative points, and so on. All criticism of you and your jokes will result in negative points. You should only return the numerical value for the action as you assess it."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{examples}"),
        ("human", "{input}")])

    llm_chain = prompt | llm
    chain_res = llm_chain.invoke({"input": f"{name}: {action}", "examples": examples}, config)
    print("Nice response: ", chain_res)
    nice_score = float(chain_res["nice_score"])

    conn = config.get("configurable", {}).get("conn")
    if not conn:
        print("No connection found in config")
        raise ValueError("No connection found in config")
    try:
        with conn._cursor() as cur:
            # Upsert the score by Name
            res = cur.execute("INSERT INTO naughty_nice (name, nice_meter) VALUES (%s, %s) ON CONFLICT (name) DO UPDATE SET nice_meter = naughty_nice.nice_meter + EXCLUDED.nice_meter, updates = naughty_nice.updates + 1 RETURNING *", (name, nice_score))
            print("Upsert result: ", res)
    except Exception as e:
        print("Error: ", e)
        conn.rollback()
        raise e

    return "Action registered!"

tools = [check_naughty_list, register_naughty_or_nice]
tool_node = ToolNode(tools)

llm_with_tools = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def santa(state: State, config: RunnableConfig):
    response = llm_with_tools.invoke(
            [("system", system_prompt), *state["messages"]],
            config)
    return { "messages": [response]}

graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("santa", santa)
graph_builder.add_node("tools", tool_node)

# Add edges
graph_builder.add_edge(START, "santa")
graph_builder.add_conditional_edges("santa", tools_condition)
graph_builder.add_edge("tools", "santa")

def get_response(graph: CompiledStateGraph, user_input: str, thread_id: str, checkpointer: PostgresSaver):
    config = { "configurable": { "thread_id": thread_id, "conn": checkpointer } }
    print("Config: ", config)
    return graph.stream(
            { "messages": [("user", user_input)] },
            config,
            stream_mode="messages")

def transform_response_to_text(response_generator):
    """
    Transform the AI message chunks from get_response into plain text.
    """
    for message, metadata in response_generator:
        if metadata["langgraph_node"] == "santa":
            yield message.content # Extract and yield plain text

def run_graph(graph: CompiledStateGraph, checkpointer: PostgresSaver):
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(random.randint(0, 1000000))

    config = { "configurable": { "thread_id": st.session_state.thread_id, "conn": checkpointer } }
    print("Thread ID: ", st.session_state.thread_id)

    state = graph.get_state(config).values

    if not "messages" in state or len(state["messages"]) == 0:
        graph.update_state(config, { "messages": [greeting_msg] })
        state = graph.get_state(config).values


    if "messages" in state:
        for message in state["messages"]:
            if message.content and isinstance(message, AIMessage):
                with st.chat_message("Julenissen"):
                    st.write(message.content)
            elif message.content and isinstance(message, HumanMessage):
                with st.chat_message("Deg"):
                    st.write(message.content)

    user_input = st.chat_input("Write your message to Santa here:")
    if user_input is not None and user_input != "":
        st.session_state.query = HumanMessage(user_input)

        with st.chat_message("You"):
            st.markdown(user_input)
            st.write("")

        with st.chat_message("Santa"):
            response_generator = get_response(graph, user_input, st.session_state.thread_id, checkpointer)
            transformed_response = transform_response_to_text(response_generator)
            st.write_stream(transformed_response)

def create_topscores(checkpointer: PostgresSaver):
    with checkpointer._cursor() as cur:
        cur.execute("SELECT name, nice_meter FROM naughty_nice where nice_meter > 0 ORDER BY nice_meter DESC LIMIT 10")
        nice_scores = cur.fetchall()
        print("Nice scores: ", nice_scores)
        cur.execute("SELECT name, nice_meter FROM naughty_nice where nice_meter < 0 ORDER BY nice_meter ASC LIMIT 10")
        naughty_scores = cur.fetchall()
        print("Naughty scores: ", naughty_scores)

    with st.sidebar:
        st.markdown("## Top 10 nice names")
        if len(nice_scores) == 0:
            st.markdown("__No names on the nice list yet!__")

        i = 1
        for row in nice_scores:
            st.markdown(f"**{i}) {row['name']}** ({row['nice_meter']} points)")
            i += 1

        st.markdown("## Top 10 naughty names")
        if len(naughty_scores) == 0:
            st.markdown("__No names on the naughty list yet!__")
        i = 1
        for row in naughty_scores:
            st.markdown(f"**{i}) {row['name']}** ({row['nice_meter']} points)")
            i += 1

        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.html('<hr style="border-top: 1px solid #ccc;margin-bottom:0;">')
        st.markdown("""
*Made by Øystein Malt*

[![Kraftlauget](https://images.squarespace-cdn.com/content/v1/610a80b3adce6b72205d4788/ebb92466-5536-4c00-bfea-a30481d5a3ac/Web-logo_500px.png?format=1500w)](https://kraftlauget.no)""")

        st.markdown("Don't miss [the christmas calendar](https://julekalender.kraftlauget.no/2024/luke/10) that explains how the digital santa was made!")

def run():
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()
        with checkpointer._cursor() as cur:
            cur.execute("CREATE TABLE IF NOT EXISTS naughty_nice (name TEXT PRIMARY KEY, nice_meter INT, updates INT DEFAULT 1)")

        create_topscores(checkpointer)

        graph = graph_builder.compile(checkpointer=checkpointer)
        run_graph(graph, checkpointer)

run()
