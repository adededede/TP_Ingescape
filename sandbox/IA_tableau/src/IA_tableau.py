#!/usr/bin/env -P /usr/bin:/usr/local/bin python3 -B
# coding: utf-8

#
#  IA_tableau.py
#  IA_tableau version 1.0
#  Created by Ingenuity i/o on 2023/11/16
#
# ia chargé de générer des images
#
import ingescape as igs


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class IATableau(metaclass=Singleton):
    def __init__(self):
        # inputs
        self.actionI = None

        # outputs
        self._imagesO = None

    # outputs
    @property
    def imagesO(self):
        return self._imagesO

    @imagesO.setter
    def imagesO(self, value):
        self._imagesO = value
        if self._imagesO is not None:
            igs.output_set_data("images", value)

    # services
    def creation_tableau(self, sender_agent_name, sender_agent_uuid, couleur):
        pass
        # add code here if needed


