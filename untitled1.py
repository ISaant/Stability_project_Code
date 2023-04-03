#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 16:39:42 2023

@author: isaac
"""

class Pet:
    
    def __init__(self,name,age):
        self.name = name
        self.age = age
        
    def show(self):
        print(f"I am {self.name} and I am {self.age} years old")
        
        
class Cat(Pet):
    def __init__ (self, name, age, color):
        super().__init__(name, age)
        self.color = color
    
    def speak(self):
        print('Meow')
    
    # def show(self):
    #     print(f"I am {self.name}, I am {self.age} years old and I am {self.color}")
p = Pet('Tim', 6)
p.show()

cat=Cat('Timmy', 7,'Gray')
cat.show()
cat.speak()