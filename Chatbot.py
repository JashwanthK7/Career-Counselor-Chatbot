#!/usr/bin/env python
# coding: utf-8

# In[1]:


import io
import random
import string # to process standard python strings
import warnings
import numpy as np
import pandas as pd
import counselor
import streamlit as st
import imagify
from PIL import Image
from bokeh.plotting import figure, output_file, show
import math
from bokeh.palettes import Greens
from bokeh.transform import cumsum
from bokeh.models import LabelSet, ColumnDataSource
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


# In[2]:


import nltk
from nltk.stem import WordNetLemmatizer


# In[3]:


f=open('chatbot.txt','r',errors = 'ignore')
raw=f.read()
raw = raw.lower()# converts to lowercase


# In[4]:


sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words


# In[5]:


lemmer = nltk.stem.WordNetLemmatizer()
n=1
#WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# In[6]:


GREETING_INPUTS = ("hello", "hi", "greetings", "what's up","hey")
GREETING_RESPONSES = "hello"
def greeting(sentence):
 
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return GREETING_RESPONSES


# In[7]:


def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you. For personalsied questions, suggest you to take the personality test."
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response


# In[8]:


qvals = {"Select an Option": 0, "Strongly Agree": 5, "Agree": 4, "Neutral": 3, "Disagree": 2,
             "Strongly Disagree": 1}
st.title("CounselBot")
banner=Image.open("img/21.png")
st.image(banner, use_column_width=True)
st.write("Hi! I'm CounselBot, your personal career counseling bot. Ask your queries below. If and when you are ready to take our personality test, type \"start my test\" and you're good to go!")


# In[ ]:


flag=True
j=0
while (flag==True and j<10):
    user_response = st.text_input("You : ",key=j)
    user_response = user_response.lower()
    if user_response in GREETING_INPUTS:
        ans = greeting(user_response)
    elif(user_response==''):
        ans = ''
    elif(user_response!=''):
        if(user_response=='start my test'):
            flag=False
            ans=''
        else:
            ans = response(user_response)
            sent_tokens.remove(user_response)

    st.write(ans)


# In[ ]:


if(flag==False):
        #x=start_test()
        #st.text_area("confirm", value="starting test", height=100, max_chars=None)
    st.title("PERSONALITY TEST:")
        #st.write("Would you like to begin with the test?")
    kr = st.selectbox("Would you like to begin with the test?", ["Select an Option", "Yes", "No"])
    if (kr == "Yes"):
        kr1 = st.selectbox("Select level of education",
                               ["Select an Option", "Grade 10", "Grade 12", "Undergraduate","ECE Undergraduate"])

            #####################################  GRADE 10  ###########################################

        if(kr1=="Grade 10"):
            lis = []
            if (kr == "Yes"):
                st.header("Question 1")
                st.write("I find writing programs for computer applications interesting")
                n = imagify.imageify(n)
                inp = st.selectbox("",
                                       ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                        "Strongly Disagree"],
                                       key='1')
                if ((inp != "Select an Option")):
                    lis.append(qvals[inp])
                    st.header("Question 2")
                    st.write("I can understand mathematical problems with ease")
                    n = imagify.imageify(n)
                    inp2 = st.selectbox("", ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                 "Strongly Disagree"], key='2')

                    if (inp2 != "Select an Option"):
                        lis.append(qvals[inp2])
                        st.header("Question 3")
                        st.write("Learning about the existence of individual chemical components is interesting")
                        n = imagify.imageify(n)
                        inp3 = st.selectbox("", ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                     "Strongly Disagree"], key='3')
                        if (inp3 != "Select an Option"):
                            lis.append(qvals[inp3])
                            st.header("Question 4")
                            st.write("The way plants and animals thrive gets me curious")
                            n = imagify.imageify(n)
                            inp4 = st.selectbox("",
                                                    ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                     "Strongly Disagree"], key='4')
                            if (inp4 != "Select an Option"):
                                lis.append(qvals[inp4])
                                st.header("Question 5")
                                st.write("Studying about the way fundamental constituents of the universe interact with each other is fascinating")
                                n = imagify.imageify(n)
                                inp5 = st.selectbox("",
                                                        ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                         "Disagree",
                                                         "Strongly Disagree"], key='5')
                                if (inp5 != "Select an Option"):
                                    lis.append(qvals[inp5])
                                    st.header("Question 6")
                                    st.write(
                                            "Accounting and business management is my cup of tea")
                                    n = imagify.imageify(n)
                                    inp6 = st.selectbox("",
                                                            ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                             "Disagree",
                                                             "Strongly Disagree"], key='6')
                                    if (inp6 != "Select an Option"):
                                        lis.append(qvals[inp6])
                                        st.header("Question 7")
                                        st.write(
                                                "I would like to know more about human behaviour, relations and patterns of thinking")
                                        n = imagify.imageify(n)
                                        inp7 = st.selectbox("",
                                                                ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                                 "Disagree",
                                                                 "Strongly Disagree"], key='7')
                                        if (inp7 != "Select an Option"):
                                            lis.append(qvals[inp7])
                                            st.header("Question 8")
                                            st.write(
                                                    "I find the need to be aware of stories from the past.")
                                            n = imagify.imageify(n)
                                            inp8 = st.selectbox("",
                                                                    ["Select an Option", "Strongly Agree", "Agree",
                                                                     "Neutral",
                                                                     "Disagree",
                                                                     "Strongly Disagree"], key='8')
                                            if (inp8 != "Select an Option"):
                                                lis.append(qvals[inp8])
                                                st.header("Question 9")
                                                st.write(
                                                        "I see myself as a sportsperson/professional trainer")
                                                n = imagify.imageify(n)
                                                inp9 = st.selectbox("",
                                                                        ["Select an Option", "Strongly Agree", "Agree",
                                                                         "Neutral",
                                                                         "Disagree",
                                                                         "Strongly Disagree"], key='9')
                                                if (inp9 != "Select an Option"):
                                                    lis.append(qvals[inp9])
                                                    st.header("Question 10")
                                                    st.write(
                                                            "I enjoy creating works of art")
                                                    n = imagify.imageify(n)
                                                    inp10 = st.selectbox("",
                                                                             ["Select an Option", "Strongly Agree", "Agree",
                                                                              "Neutral",
                                                                              "Disagree",
                                                                              "Strongly Disagree"], key='10')
                                                    if (inp10 != "Select an Option"):
                                                        lis.append(qvals[inp10])
                                                        st.success("Test Completed")
                                                            #st.write(lis)
                                                        st.title("RESULTS:")
                                                        df = pd.read_csv(r"Subjects.csv")

                                                        input_list = lis

                                                        subjects = {1: "Computers",
                                                                        2: "Mathematics",
                                                                        3: "Chemistry",
                                                                        4: "Biology",
                                                                        5: "Physics",
                                                                        6: "Commerce",
                                                                        7: "Psychology",
                                                                        8: "History",
                                                                        9: "Physical Education",
                                                                        10: "Design"}

                                                        def output(listofanswers):
                                                            class my_dictionary(dict):
                                                                def __init__(self):
                                                                    self = dict()

                                                                def add(self, key, value):
                                                                    self[key] = value

                                                            ques = my_dictionary()

                                                            for i in range(0, 10):
                                                                ques.add(i, input_list[i])

                                                            all_scores = []

                                                            for i in range(9):
                                                                all_scores.append(ques[i] / 5)

                                                            li = []

                                                            for i in range(len(all_scores)):
                                                                li.append([all_scores[i], i])
                                                            li.sort(reverse=True)
                                                            sort_index = []
                                                            for x in li:
                                                                sort_index.append(x[1] + 1)
                                                            all_scores.sort(reverse=True)

                                                            a = sort_index[0:5]
                                                            b = all_scores[0:5]
                                                            s = sum(b)
                                                            d = list(map(lambda x: x * (100 / s), b))

                                                            return a, d

                                                        l, data = output(input_list)

                                                        out = []
                                                        for i in range(0, 5):
                                                            n = l[i]
                                                            c = subjects[n]
                                                            out.append(c)

                                                        output_file("pie.html")

                                                        graph = figure(title="Recommended subjects", height=500,
                                                                           width=500)
                                                        radians = [math.radians((percent / 100) * 360) for percent
                                                                       in data]

                                                        start_angle = [math.radians(0)]
                                                        prev = start_angle[0]
                                                        for i in radians[:-1]:
                                                            start_angle.append(i + prev)
                                                            prev = i + prev

                                                        end_angle = start_angle[1:] + [math.radians(0)]

                                                        x = 0
                                                        y = 0

                                                        radius = 0.8

                                                        color = Greens[len(out)]
                                                        graph.xgrid.visible = False
                                                        graph.ygrid.visible = False
                                                        graph.xaxis.visible = False
                                                        graph.yaxis.visible = False

                                                        for i in range(len(out)):
                                                            graph.wedge(x, y, radius,
                                                                            start_angle=start_angle[i],
                                                                            end_angle=end_angle[i],
                                                                            color=color[i],
                                                                            legend_label=out[i] + "-" + str(
                                                                                round(data[i])) + "%")

                                                        graph.add_layout(graph.legend[0], 'right')
                                                        st.bokeh_chart(graph, use_container_width=True)
                                                        labels = LabelSet(x='text_pos_x', y='text_pos_y',
                                                                                text='percentage', level='glyph',
                                                                                angle=0, render_mode='canvas')
                                                        graph.add_layout(labels)

                                                        st.header('More information on the subjects')
                                                            # We'll be using a csv file for that
                                                        for i in range(0, 5):
                                                            st.subheader(subjects[int(l[i])])
                                                            st.write(df['about'][int(l[i]) - 1])

                                                        st.header('Choice of Degrees')
                                                            # We'll be using a csv file for that
                                                        for i in range(0, 5):
                                                            st.subheader(subjects[int(l[i])])
                                                            st.write(df['further career'][int(l[i]) - 1])


        ##########################################  GRADE 12  ########################################################

        elif (kr1 == "Grade 12"):
            lis = []
            st.header("Question 1")
            st.write("I enjoy debating and negotiating issues in public")
            n = imagify.imageify(n)
            inp = st.selectbox("",
                                   ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                    "Strongly Disagree"],
                                   key='1')
            if ((inp != "Select an Option")):
                lis.append(qvals[inp])
                st.header("Question 2")
                st.write("Studying the anatomy of the human body and giving first aid to people is something I'm always looking forward to")
                n = imagify.imageify(n)
                inp2 = st.selectbox("", ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                             "Strongly Disagree"], key='2')

                if (inp2 != "Select an Option"):
                    lis.append(qvals[inp2])
                    st.header("Question 3")
                    st.write("I can lead a team and easily manage projects")
                    n = imagify.imageify(n)
                    inp3 = st.selectbox("", ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                 "Strongly Disagree"], key='3')
                    if (inp3 != "Select an Option"):
                        lis.append(qvals[inp3])
                        st.header("Question 4")
                        st.write("Working with tools, equipment, and machinery is enjoyable")
                        n = imagify.imageify(n)
                        inp4 = st.selectbox("",
                                                ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                 "Strongly Disagree"], key='4')
                        if (inp4 != "Select an Option"):
                            lis.append(qvals[inp4])
                            st.header("Question 5")
                            st.write(
                                    "Budgeting, costing and estimating for a business isn't exhausting")
                            n = imagify.imageify(n)
                            inp5 = st.selectbox("",
                                                    ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                     "Disagree",
                                                     "Strongly Disagree"], key='5')
                            if (inp5 != "Select an Option"):
                                lis.append(qvals[inp5])
                                st.header("Question 6")
                                st.write(
                                        "I can see myself taking part in competitive sporting events to become a professional")
                                n = imagify.imageify(n)
                                inp6 = st.selectbox("",
                                                        ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                         "Disagree",
                                                         "Strongly Disagree"], key='6')
                                if (inp6 != "Select an Option"):
                                    lis.append(qvals[inp6])
                                    st.header("Question 7")
                                    st.write(
                                            "I don't burn out while doing translations, reading and correcting language")
                                    n = imagify.imageify(n)
                                    inp7 = st.selectbox("",
                                                            ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                             "Disagree",
                                                             "Strongly Disagree"], key='7')
                                    if (inp7 != "Select an Option"):
                                        lis.append(qvals[inp7])
                                        st.header("Question 8")
                                        st.write(
                                                "I would love to act in or direct a play or film")
                                        n = imagify.imageify(n)
                                        inp8 = st.selectbox("",
                                                                ["Select an Option", "Strongly Agree", "Agree",
                                                                 "Neutral",
                                                                 "Disagree",
                                                                 "Strongly Disagree"], key='8')
                                        if (inp8 != "Select an Option"):
                                            lis.append(qvals[inp8])
                                            st.header("Question 9")
                                            st.write(
                                                    "Making sketches of people or landscapes is a hobby I see as a career")
                                            n = imagify.imageify(n)
                                            inp9 = st.selectbox("",
                                                                    ["Select an Option", "Strongly Agree", "Agree",
                                                                     "Neutral",
                                                                     "Disagree",
                                                                     "Strongly Disagree"], key='9')
                                            if (inp9 != "Select an Option"):
                                                lis.append(qvals[inp9])
                                                st.header("Question 10")
                                                st.write(
                                                        "I can easily work with numbers and calculations most of the time")
                                                n = imagify.imageify(n)
                                                inp10 = st.selectbox("",
                                                                         ["Select an Option", "Strongly Agree", "Agree",
                                                                          "Neutral",
                                                                          "Disagree",
                                                                          "Strongly Disagree"], key='10')
                                                if (inp10 != "Select an Option"):
                                                    lis.append(qvals[inp10])
                                                    st.header("Question 11")
                                                    st.write(
                                                            "I enjoy doing clerical work i.e. filing, counting stock and issuing receipts")
                                                    n = imagify.imageify(n)
                                                    inp11 = st.selectbox("",
                                                                             ["Select an Option", "Strongly Agree",
                                                                              "Agree",
                                                                              "Neutral",
                                                                              "Disagree",
                                                                              "Strongly Disagree"], key='11')
                                                    if (inp11 != "Select an Option"):
                                                        lis.append(qvals[inp11])
                                                        st.header("Question 12")
                                                        st.write(
                                                                "I love studying the culture and life style of human societies")
                                                        n = imagify.imageify(n)
                                                        inp12 = st.selectbox("",
                                                                                 ["Select an Option", "Strongly Agree",
                                                                                  "Agree",
                                                                                  "Neutral",
                                                                                  "Disagree",
                                                                                  "Strongly Disagree"], key='12')
                                                        if (inp12 != "Select an Option"):
                                                            lis.append(qvals[inp12])
                                                            st.header("Question 13")
                                                            st.write(
                                                                    "Teaching children and young people is something I see myself doing on a daily basis")
                                                            n = imagify.imageify(n)
                                                            inp13 = st.selectbox("",
                                                                                     ["Select an Option",
                                                                                      "Strongly Agree", "Agree",
                                                                                      "Neutral",
                                                                                      "Disagree",
                                                                                      "Strongly Disagree"], key='13')
                                                            if (inp13 != "Select an Option"):
                                                                lis.append(qvals[inp13])
                                                                st.header("Question 14")
                                                                st.write(
                                                                        "I won't have a problem persevering in the army or police force")
                                                                n = imagify.imageify(n)
                                                                inp14 = st.selectbox("",
                                                                                         ["Select an Option",
                                                                                          "Strongly Agree", "Agree",
                                                                                          "Neutral",
                                                                                          "Disagree",
                                                                                          "Strongly Disagree"],
                                                                                         key='14')
                                                                if (inp14 != "Select an Option"):
                                                                    lis.append(qvals[inp14])
                                                                    st.header("Question 15")
                                                                    st.write(
                                                                            "Introducing consumers to new products and convincing them to buy the same is something that comes with ease")
                                                                    n = imagify.imageify(n)
                                                                    inp15 = st.selectbox("",
                                                                                             ["Select an Option",
                                                                                              "Strongly Agree", "Agree",
                                                                                              "Neutral",
                                                                                              "Disagree",
                                                                                              "Strongly Disagree"],
                                                                                             key='15')
                                                                    if (inp15 != "Select an Option"):
                                                                        lis.append(qvals[inp10])
                                                                        st.success("Test Completed")
                                                                            #st.write(lis)
                                                                        st.title("RESULTS:")
                                                                        df = pd.read_csv(r"Graduate.csv")

                                                                        input_list = lis

                                                                        streams = {1: "Law",
                                                                                       2: "Healthcare",
                                                                                       3: "Management",
                                                                                       4: "Engineering",
                                                                                       5: "Finance",
                                                                                       6: "Sports",
                                                                                       7: "Language and communication",
                                                                                       8: "Performing Arts",
                                                                                       9: "Applied and Visual arts",
                                                                                       10: "Science and math",
                                                                                       11: "Clerical and secretarial",
                                                                                       12: "Social Science",
                                                                                       13: "Education and Social Support",
                                                                                       14: "Armed Forces",
                                                                                       15: "Marketing and sales"}

                                                                        def output(listofanswers):
                                                                            class my_dictionary(dict):
                                                                                def __init__(self):
                                                                                    self = dict()

                                                                                def add(self, key, value):
                                                                                    self[key] = value

                                                                            ques = my_dictionary()

                                                                            for i in range(0, 15):
                                                                                ques.add(i, input_list[i])

                                                                            all_scores = []

                                                                            for i in range(14):
                                                                                all_scores.append(ques[i] / 5)

                                                                            li = []

                                                                            for i in range(len(all_scores)):
                                                                                li.append([all_scores[i], i])
                                                                            li.sort(reverse=True)
                                                                            sort_index = []
                                                                            for x in li:
                                                                                sort_index.append(x[1] + 1)
                                                                            all_scores.sort(reverse=True)

                                                                            a = sort_index[0:5]
                                                                            b = all_scores[0:5]
                                                                            s = sum(b)
                                                                            d = list(
                                                                                map(lambda x: x * (100 / s), b))

                                                                            return a, d

                                                                        l, data = output(input_list)

                                                                        out = []
                                                                        for i in range(0, 5):
                                                                            n = l[i]
                                                                            c = streams[n]
                                                                            out.append(c)

                                                                        output_file("pie.html")

                                                                        graph = figure(title="Recommended fields",
                                                                                           height=500, width=500)
                                                                        radians = [
                                                                                math.radians((percent / 100) * 360) for
                                                                                percent in data]

                                                                        start_angle = [math.radians(0)]
                                                                        prev = start_angle[0]
                                                                        for i in radians[:-1]:
                                                                            start_angle.append(i + prev)
                                                                            prev = i + prev

                                                                        end_angle = start_angle[1:] + [
                                                                            math.radians(0)]

                                                                        x = 0
                                                                        y = 0

                                                                        radius = 0.8

                                                                        color = Greens[len(out)]
                                                                        graph.xgrid.visible = False
                                                                        graph.ygrid.visible = False
                                                                        graph.xaxis.visible = False
                                                                        graph.yaxis.visible = False

                                                                        for i in range(len(out)):
                                                                            graph.wedge(x, y, radius,
                                                                                            start_angle=start_angle[i],
                                                                                            end_angle=end_angle[i],
                                                                                            color=color[i],
                                                                                            legend_label=out[
                                                                                                             i] + "-" + str(
                                                                                                round(data[i])) + "%")

                                                                        graph.add_layout(graph.legend[0],
                                                                                                'right')
                                                                        st.bokeh_chart(graph,
                                                                                            use_container_width=True)
                                                                        labels = LabelSet(x='text_pos_x',
                                                                                                y='text_pos_y',
                                                                                                text='percentage',
                                                                                                level='glyph',
                                                                                                angle=0,
                                                                                                render_mode='canvas')
                                                                        graph.add_layout(labels)

                                                                        st.header(
                                                                                'More information on the fields')
                                                                            # We'll be using a csv file for that
                                                                        for i in range(0, 5):
                                                                            st.subheader(streams[int(l[i])])
                                                                            st.write(df['About'][int(l[i]) - 1])
                                                                           
                                                                        

            ######################################  UNDERGRADUATE ##########################################

        elif (kr1 == "Undergraduate"):
            lis = []
            if (kr == "Yes"):
                st.header("Question 1")
                st.write("I can be the person who handles all aspects of information security and protects the virtual data resources of a company")
                n = imagify.imageify(n)
                inp = st.selectbox("",
                                       ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                        "Strongly Disagree"],
                                       key='1')
                if ((inp != "Select an Option")):
                    lis.append(qvals[inp])
                    st.header("Question 2")
                    st.write("I enjoy studying business and information requirements of an organisation and using this data to develop processes that help achieve strategic goals.")
                    n = imagify.imageify(n)
                    inp2 = st.selectbox("", ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                 "Strongly Disagree"], key='2')

                    if (inp2 != "Select an Option"):
                        lis.append(qvals[inp2])
                        st.header("Question 3")
                        st.write("I can assess a problem and design a brand new system or improve an existing system to make it better and more efficient. ")
                        n = imagify.imageify(n)
                        inp3 = st.selectbox("", ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                     "Strongly Disagree"], key='3')
                        if (inp3 != "Select an Option"):
                            lis.append(qvals[inp3])
                            st.header("Question 4")
                            st.write("Designing, developing, modifying, editing and working with databases and large datasets is my cup of tea")
                            n = imagify.imageify(n)
                            inp4 = st.selectbox("",
                                                    ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                     "Strongly Disagree"], key='4')
                            if (inp4 != "Select an Option"):
                                lis.append(qvals[inp4])
                                st.header("Question 5")
                                st.write(
                                        "I can mine data using BI software tools, compare, visualize and communicate the results with ease")
                                n = imagify.imageify(n)
                                inp5 = st.selectbox("",
                                                        ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                         "Disagree",
                                                         "Strongly Disagree"], key='5')
                                if (inp5 != "Select an Option"):
                                    lis.append(qvals[inp5])
                                    st.header("Question 6")
                                    st.write(
                                            "Implementing and providing support for Microsoft's Dynamics CRM is a skill I possess")
                                    n = imagify.imageify(n)
                                    inp6 = st.selectbox("",
                                                            ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                             "Disagree",
                                                             "Strongly Disagree"], key='6')
                                    if (inp6 != "Select an Option"):
                                        lis.append(qvals[inp6])
                                        st.header("Question 7")
                                        st.write(
                                                "I can be innovative and creative when it comes to making user-friendly mobile applications")
                                        n = imagify.imageify(n)
                                        inp7 = st.selectbox("",
                                                                ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                                 "Disagree",
                                                                 "Strongly Disagree"], key='7')
                                        if (inp7 != "Select an Option"):
                                            lis.append(qvals[inp7])
                                            st.header("Question 8")
                                            st.write(
                                                    "I can perform well in a varied discipline, combining aspects of psychology, business, market research, design, and technology.")
                                            n = imagify.imageify(n)
                                            inp8 = st.selectbox("",
                                                                    ["Select an Option", "Strongly Agree", "Agree",
                                                                     "Neutral",
                                                                     "Disagree",
                                                                     "Strongly Disagree"], key='8')
                                            if (inp8 != "Select an Option"):
                                                lis.append(qvals[inp8])
                                                st.header("Question 9")
                                                st.write(
                                                        "I am responsible enough to maintain the quality systems, such as laboratory control and document control and training, to ensure control of the manufacturing process.")
                                                n = imagify.imageify(n)
                                                inp9 = st.selectbox("",
                                                                        ["Select an Option", "Strongly Agree", "Agree",
                                                                         "Neutral",
                                                                         "Disagree",
                                                                         "Strongly Disagree"], key='9')
                                                if (inp9 != "Select an Option"):
                                                    lis.append(qvals[inp9])
                                                    st.header("Question 10")
                                                    st.write(
                                                            "Be it front-end or back-end, I would love designing and developing websites more than anything else")
                                                    n = imagify.imageify(n)
                                                    inp10 = st.selectbox("",
                                                                             ["Select an Option", "Strongly Agree", "Agree",
                                                                              "Neutral",
                                                                              "Disagree",
                                                                              "Strongly Disagree"], key='10')
                                                    if (inp10 != "Select an Option"):
                                                        lis.append(qvals[inp10])
                                                        st.success("Test Completed")
                                                            #st.write(lis)

                                                        st.title("RESULTS:")
                                                        df = pd.read_csv(r'Occupations.csv', encoding= 'windows-1252')

                                                        input_list = lis

                                                        professions = {1: "Systems Security Administrator",
                                                                        2: "Business Systems Analyst",
                                                                        3: "Software Systems Engineer",
                                                                        4: "Database Developer",
                                                                        5: "Business Intelligence Analyst",
                                                                        6: "CRM Technical Developer",
                                                                        7: "Mobile Applications Developer",
                                                                        8: "UX Designer",
                                                                        9: "Quality Assurance Associate",
                                                                        10: "Web Developer"}

                                                        def output(listofanswers):
                                                            class my_dictionary(dict):
                                                                def __init__(self):
                                                                    self = dict()

                                                                def add(self, key, value):
                                                                    self[key] = value

                                                            ques = my_dictionary()

                                                            for i in range(0, 10):
                                                                ques.add(i, input_list[i])

                                                            all_scores = []

                                                            for i in range(9):
                                                                all_scores.append(ques[i] / 5)

                                                            li = []

                                                            for i in range(len(all_scores)):
                                                                li.append([all_scores[i], i])
                                                            li.sort(reverse=True)
                                                            sort_index = []
                                                            for x in li:
                                                                sort_index.append(x[1] + 1)
                                                            all_scores.sort(reverse=True)

                                                            a = sort_index[0:5]
                                                            b = all_scores[0:5]
                                                            s = sum(b)
                                                            d = list(map(lambda x: x * (100 / s), b))

                                                            return a, d

                                                        l, data = output(input_list)

                                                        out = []
                                                        for i in range(0, 5):
                                                            n = l[i]
                                                            c = professions[n]
                                                            out.append(c)

                                                        output_file("pie.html")

                                                        graph = figure(title="Recommended professions", height=500,
                                                                           width=500)
                                                        radians = [math.radians((percent / 100) * 360) for percent
                                                                       in data]

                                                        start_angle = [math.radians(0)]
                                                        prev = start_angle[0]
                                                        for i in radians[:-1]:
                                                            start_angle.append(i + prev)
                                                            prev = i + prev

                                                        end_angle = start_angle[1:] + [math.radians(0)]

                                                        x = 0
                                                        y = 0

                                                        radius = 0.8

                                                        color = Greens[len(out)]
                                                        graph.xgrid.visible = False
                                                        graph.ygrid.visible = False
                                                        graph.xaxis.visible = False
                                                        graph.yaxis.visible = False

                                                        for i in range(len(out)):
                                                            graph.wedge(x, y, radius,
                                                                            start_angle=start_angle[i],
                                                                            end_angle=end_angle[i],
                                                                            color=color[i],
                                                                            legend_label=out[i] + "-" + str(
                                                                                round(data[i])) + "%")

                                                        graph.add_layout(graph.legend[0], 'right')
                                                        st.bokeh_chart(graph, use_container_width=True)
                                                        labels = LabelSet(x='text_pos_x', y='text_pos_y',
                                                                                text='percentage', level='glyph',
                                                                                angle=0, render_mode='canvas')
                                                        graph.add_layout(labels)
                                                        st.header('More information on the professions')
                                                            # We'll be using a csv file for that
                                                        for i in range(0, 5):
                                                            st.subheader(professions[int(l[i])])
                                                            st.write(df['Information'][int(l[i]) - 1])
                                                            
                                                            
        elif (kr1 == "ECE Undergraduate"):
            lis = []
            if (kr == "Yes"):
                st.header("Question 1")
                st.write("I can be the person who can establish, maintain network performance, build net configurations and connections of a company")
                inp = st.selectbox("",
                                       ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                        "Strongly Disagree"],
                                       key='1')
                if ((inp != "Select an Option")):
                    lis.append(qvals[inp])
                    st.header("Question 2")
                    st.write("I can be the person who can write technical specifications and define the architecture of RF solutions")
                    inp2 = st.selectbox("", ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                 "Strongly Disagree"], key='2')

                    if (inp2 != "Select an Option"):
                        lis.append(qvals[inp2])
                        st.header("Question 3")
                        st.write("I can be the person who can design manufacturing processes for electronic devices")
                        inp3 = st.selectbox("", ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                     "Strongly Disagree"], key='3')
                        if (inp3 != "Select an Option"):
                            lis.append(qvals[inp3])
                            st.header("Question 4")
                            st.write("I can be the person who can develop, design, and test embedded systems like microcontrollers")
                            inp4 = st.selectbox("",
                                                    ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                     "Strongly Disagree"], key='4')
                            if (inp4 != "Select an Option"):
                                lis.append(qvals[inp4])
                                st.header("Question 5")
                                st.write(
                                        "I am interested in Integrated circuits")
                                inp5 = st.selectbox("",
                                                        ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                         "Disagree",
                                                         "Strongly Disagree"], key='5')
                                if (inp5 != "Select an Option"):
                                    lis.append(qvals[inp5])
                                    st.header("Question 6")
                                    st.write(
                                            "I can solve electronic problems associated with aircrafts")
                                    inp6 = st.selectbox("",
                                                            ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                             "Disagree",
                                                             "Strongly Disagree"], key='6')
                                    if (inp6 != "Select an Option"):
                                        lis.append(qvals[inp6])
                                        st.header("Question 7")
                                        st.write(
                                                "I can be the person who can design and develop electronic control systems for vehicles")
                                        inp7 = st.selectbox("",
                                                                ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                                 "Disagree",
                                                                 "Strongly Disagree"], key='7')
                                        if (inp7 != "Select an Option"):
                                            lis.append(qvals[inp7])
                                            st.header("Question 8")
                                            st.write(
                                                    "Programming and coding firmware and software for embedded systems to facilitate device functionality and communication is my cup of tea")
                                            inp8 = st.selectbox("",
                                                                    ["Select an Option", "Strongly Agree", "Agree",
                                                                     "Neutral",
                                                                     "Disagree",
                                                                     "Strongly Disagree"], key='8')
                                            if (inp8 != "Select an Option"):
                                                lis.append(qvals[inp8])
                                                st.success("Test Completed")
                                                            #st.write(lis)

                                                st.title("RESULTS:")
                                                df = pd.read_csv(r'Ece.csv', encoding= 'windows-1252')

                                                input_list = lis

                                                professions = {1: "Network Engineer",
                                                                        2: "RF Engineer",
                                                                        3: "Design Engineer",
                                                                        4: "Embedded Systems Engineer",
                                                                        5: "VLSI Design Engineer",
                                                                        6: "Avionics Engineer",
                                                                        7: "Automotive Electronics Engineer",
                                                                        8: "Internet of Things (IoT) Engineer"}

                                                def output(listofanswers):
                                                    class my_dictionary(dict):
                                                        def __init__(self):
                                                            self = dict()

                                                        def add(self, key, value):
                                                            self[key] = value

                                                    ques = my_dictionary()

                                                    for i in range(0, 8):
                                                        ques.add(i, input_list[i])

                                                    all_scores = []

                                                    for i in range(7):
                                                        all_scores.append(ques[i] / 5)

                                                    li = []

                                                    for i in range(len(all_scores)):
                                                        li.append([all_scores[i], i])
                                                    li.sort(reverse=True)
                                                    sort_index = []
                                                    for x in li:
                                                        sort_index.append(x[1] + 1)
                                                    all_scores.sort(reverse=True)

                                                    a = sort_index[0:5]
                                                    b = all_scores[0:5]
                                                    s = sum(b)
                                                    d = list(map(lambda x: x * (100 / s), b))

                                                    return a, d

                                                l, data = output(input_list)

                                                out = []
                                                for i in range(0, 5):
                                                    n = l[i]
                                                    c = professions[n]
                                                    out.append(c)

                                                output_file("pie.html")

                                                graph = figure(title="Recommended professions", height=500,
                                                                           width=500)
                                                radians = [math.radians((percent / 100) * 360) for percent
                                                                       in data]

                                                start_angle = [math.radians(0)]
                                                prev = start_angle[0]
                                                for i in radians[:-1]:
                                                    start_angle.append(i + prev)
                                                    prev = i + prev

                                                end_angle = start_angle[1:] + [math.radians(0)]

                                                x = 0
                                                y = 0

                                                radius = 0.8

                                                color = Greens[len(out)]
                                                graph.xgrid.visible = False
                                                graph.ygrid.visible = False
                                                graph.xaxis.visible = False
                                                graph.yaxis.visible = False

                                                for i in range(len(out)):
                                                    graph.wedge(x, y, radius,
                                                                            start_angle=start_angle[i],
                                                                            end_angle=end_angle[i],
                                                                            color=color[i],
                                                                            legend_label=out[i] + "-" + str(
                                                                                round(data[i])) + "%")

                                                graph.add_layout(graph.legend[0], 'right')
                                                st.bokeh_chart(graph, use_container_width=True)
                                                labels = LabelSet(x='text_pos_x', y='text_pos_y',
                                                                                text='percentage', level='glyph',
                                                                                angle=0, render_mode='canvas')
                                                graph.add_layout(labels)
                                                st.header('More information on the professions')
                                                            # We'll be using a csv file for that
                                                for i in range(0, 5):
                                                    st.subheader(professions[int(l[i])])
                                                    st.write(df['Information'][int(l[i]) - 1])
                                                         
                                                         

