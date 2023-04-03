#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 17:08:47 2023

@author: isaac
"""

class Person:
    number_of_people = 0
    def __init__(self,name):
        self.name = name

p1 = Person('Tom')
p2 = Person('Joel')

p1.number_of_people = 8
print (p1.name, p1.number_of_people)
print (p2.name, p2.number_of_people)
Person.number_of_people = 10 
print (p1.name, p1.number_of_people)
print (p2.name, p2.number_of_people)