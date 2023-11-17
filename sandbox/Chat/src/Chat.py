#!/usr/bin/env -P /usr/bin:/usr/local/bin python3 -B
# coding: utf-8

#
#  Chat.py
#  Chat version 1.0
#  Created by Ingenuity i/o on 2023/11/16
#
# "no description"
#
import ingescape as igs


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Chat(metaclass=Singleton):
    def __init__(self):
        # inputs
        self.lastChatMessageI = None

        # outputs
        self._commandeO = None
        self._messageO = None

    # outputs
    @property
    def commandeO(self):
        return self._commandeO

    @commandeO.setter
    def commandeO(self, value):
        self._commandeO = value
        if self._commandeO is not None:
            igs.output_set_string("commande", self._commandeO)
    @property
    def messageO(self):
        return self._messageO

    @messageO.setter
    def messageO(self, value):
        self._messageO = value
        if self._messageO is not None:
            igs.output_set_string("message", self._messageO)


