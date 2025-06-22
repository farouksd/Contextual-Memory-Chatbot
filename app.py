import streamlit as st
from memory import ConversationMemory
from knowledge_graph import KnowledgeGraph
from chatbot import Chatbot
import json


memory = ConversationMemory()
kg = KnowledgeGraph()
bot = Chatbot()


st.title("🧠 Contextual Memory Chatbot")
st.markdown("This AI remembers past conversations and builds a knowledge graph!")

user_input = st.text_input("You:")
if user_input:
    
    relevant_memories = memory.get_relevant_memories(user_input)
    context = "\n".join([f"User: {m['user']}\nBot: {m['bot']}" for m in relevant_memories])
    
   
    bot_response = bot.generate_response(user_input, context)
    
    memory.add_conversation(user_input, bot_response)
    kg.update_graph(user_input + " " + bot_response)
    
    st.text_area("Bot:", value=bot_response, height=100)
    
    if relevant_memories:
        st.subheader("📜 Related Past Conversations")
        for mem in relevant_memories:
            st.write(f"**You:** {mem['user']}")
            st.write(f"**Bot:** {mem['bot']}")
            st.write("---")
    
    st.subheader("🧠 Knowledge Graph")
    net = kg.visualize()
    net.save_graph("graph.html")
    st.components.v1.html(open("graph.html", "r").read(), height=500)
