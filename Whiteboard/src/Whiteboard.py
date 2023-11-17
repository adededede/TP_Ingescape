#!/usr/bin/env -P /usr/bin:/usr/local/bin python3 -B
# coding: utf-8

#
#  Whiteboard.py
#  Whiteboard
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


class Whiteboard(metaclass=Singleton):
    def __init__(self):
        # inputs
        self.titleI = None
        self.backgroundColorI = None
        self.chatMessageI = None
        self.ui_commandI = None

        # outputs
        self._lastChatMessageO = None
        self._lastActionO = None
        self._ui_errorO = None

    # outputs
    @property
    def lastChatMessageO(self):
        return self._lastChatMessageO

    @lastChatMessageO.setter
    def lastChatMessageO(self, value):
        self._lastChatMessageO = value
        if self._lastChatMessageO is not None:
            igs.output_set_string("lastChatMessage", self._lastChatMessageO)
    @property
    def lastActionO(self):
        return self._lastActionO

    @lastActionO.setter
    def lastActionO(self, value):
        self._lastActionO = value
        if self._lastActionO is not None:
            igs.output_set_string("lastAction", self._lastActionO)
    @property
    def ui_errorO(self):
        return self._ui_errorO

    @ui_errorO.setter
    def ui_errorO(self, value):
        self._ui_errorO = value
        if self._ui_errorO is not None:
            igs.output_set_string("ui_error", self._ui_errorO)

    # services
    def chat(self, sender_agent_name, sender_agent_uuid, message):
        pass
        # add code here if needed

    def snapshot(self, sender_agent_name, sender_agent_uuid):
        pass
        # add code here if needed

    def clear(self, sender_agent_name, sender_agent_uuid):
        pass
        # add code here if needed

    def addShape(self, sender_agent_name, sender_agent_uuid, type, x, y, width, height, fill, stroke, strokeWidth):
        pass
        # add code here if needed

    def addText(self, sender_agent_name, sender_agent_uuid, text, x, y, color):
        pass
        # add code here if needed

    def addImage(self, sender_agent_name, sender_agent_uuid, base64, x, y, width, height):
        pass
        # add code here if needed

    def addImageFromUrl(self, sender_agent_name, sender_agent_uuid, url, x, y):
        pass
        # add code here if needed

    def remove(self, sender_agent_name, sender_agent_uuid, elementId):
        pass
        # add code here if needed

    def translate(self, sender_agent_name, sender_agent_uuid, elementId, dx, dy):
        pass
        # add code here if needed

    def moveTo(self, sender_agent_name, sender_agent_uuid, elementId, x, y):
        pass
        # add code here if needed

    def setStringProperty(self, sender_agent_name, sender_agent_uuid, elementId, property, value):
        pass
        # add code here if needed

    def setDoubleProperty(self, sender_agent_name, sender_agent_uuid, elementId, property, value):
        pass
        # add code here if needed

    def getElementIds(self, sender_agent_name, sender_agent_uuid):
        pass
        # add code here if needed

    def getElements(self, sender_agent_name, sender_agent_uuid):
        pass
        # add code here if needed


