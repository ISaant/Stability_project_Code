#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 08:15:10 2023

@author: isaac
"""
import os
from git import Repo

class Git:
    def __init__ (self,COMMIT_MESSAGE='comment from python script'):
        self.git_path_to_repo=os.getcwd()+'/.git'  # make sure .git folder is properly configured
        self.message=COMMIT_MESSAGE
    def git_push(self):
        try:
            repo = Repo(self.git_path_to_repo)
            # repo.git.add(update=True)
            repo.git.add(all=True)
            repo.index.commit(self.message)
            origin = repo.remote(name='origin')
            origin.push()
        except:
            print('Some error occured while pushing the code')    

    def git_pull(self):
        # try:
        repo = Repo(self.git_path_to_repo)
        origin = repo.remote(name='origin')
        origin.pull()
        # except:
        # print('Some error occured while pulling the code')