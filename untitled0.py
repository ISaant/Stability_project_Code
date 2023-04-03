#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 16:11:50 2023

@author: isaac
"""

class Student:
    
    def __init__(self,name,age,grade):
        self.name=name
        self.age=age
        self.grade=grade
    
    def get_grade(self):
        return self.grade 
    
class Course:
    
    def __init__(self, name, max_student):
        self.name = name
        self.max_student = max_student
        self.students=[]
        
    def add_student(self,student):
        if len(self.students) < self.max_student:
            self.students.append(student)  
    
    def get_average_grade(self):
        value = 0
        for student in self.students:
            value += student.grade
            
        return value/len(self.students)

s1=Student('Tim', 19, 95)
s2=Student('Jim', 19, 75)
s3=Student('Bill', 19, 65)

course = Course('Science', 2)
course.add_student(s1)
course.add_student(s2)

print (course.get_average_grade())
    