#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
number = np.random.randint(1,100)

def game_core(number):
    count = 0
    nach = 0
    kon =100
    predict = np.random.randint(0,100)
    while number != predict:
        count+=1
        if number > predict:
            nach = predict
            predict = np.random.randint(nach,kon)  
        elif number < predict:
            kon = predict
            predict = np.random.randint(nach,kon)
    return(count)

def score_game(game_core):
    count_ls = []
    np.random.seed(1)  
    random_array = np.random.randint(1, 101, size=(150))
    for number in random_array:
        count_ls.append(game_core(number))
    score = int(np.mean(count_ls))
    print(f"Ваш алгоритм угадывает число в среднем за {score} попыток")
    return(score)

score_game(game_core)

