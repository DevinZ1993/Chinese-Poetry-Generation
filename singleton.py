#! /usr/bin/env python3
#-*- coding:utf-8 -*-

# Reference: https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa.

class Singleton(object):
    
    _instance = None

    def __new__(class_, *args, **kwargs):
        if not class_._instance:
            class_._instance = object.__new__(class_, *args, **kwargs)
        return class_._instance

