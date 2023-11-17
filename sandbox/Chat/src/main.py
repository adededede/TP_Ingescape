#!/usr/bin/env -P /usr/bin:/usr/local/bin python3 -B
# coding: utf-8

#
#  main.py
#  Chat version 1.0
#  Created by Ingenuity i/o on 2023/11/16
#
# "no description"
#

import signal
import getopt
import time
from pathlib import Path
import traceback
import sys
import re

from Chat import *

port = 15670
agent_name = "Chat"
device = None
verbose = False
is_interrupted = False

short_flag = "hvip:d:n:"
long_flag = ["help", "verbose", "interactive_loop", "port=", "device=", "name="]

ingescape_path = Path("~/Documents/Ingescape").expanduser()


def print_usage():
    print("Usage example: ", agent_name, " --verbose --port 5670 --device device_name")
    print("\nthese parameters have default value (indicated here above):")
    print("--verbose : enable verbose mode in the application (default is disabled)")
    print("--port port_number : port used for autodiscovery between agents (default: 31520)")
    print("--device device_name : name of the network device to be used (useful if several devices available)")
    print("--name agent_name : published name for this agent (default: ", agent_name, ")")
    print("--interactive_loop : enables interactive loop to pass commands in CLI (default: false)")


def print_usage_help():
    print("Available commands in the terminal:")
    print("	/quit : quits the agent")
    print("	/help : displays this message")

def return_iop_value_type_as_str(value_type):
    if value_type == igs.INTEGER_T:
        return "Integer"
    elif value_type == igs.DOUBLE_T:
        return "Double"
    elif value_type == igs.BOOL_T:
        return "Bool"
    elif value_type == igs.STRING_T:
        return "String"
    elif value_type == igs.IMPULSION_T:
        return "Impulsion"
    elif value_type == igs.DATA_T:
        return "Data"
    else:
        return "Unknown"

def return_event_type_as_str(event_type):
    if event_type == igs.PEER_ENTERED:
        return "PEER_ENTERED"
    elif event_type == igs.PEER_EXITED:
        return "PEER_EXITED"
    elif event_type == igs.AGENT_ENTERED:
        return "AGENT_ENTERED"
    elif event_type == igs.AGENT_UPDATED_DEFINITION:
        return "AGENT_UPDATED_DEFINITION"
    elif event_type == igs.AGENT_KNOWS_US:
        return "AGENT_KNOWS_US"
    elif event_type == igs.AGENT_EXITED:
        return "AGENT_EXITED"
    elif event_type == igs.AGENT_UPDATED_MAPPING:
        return "AGENT_UPDATED_MAPPING"
    elif event_type == igs.AGENT_WON_ELECTION:
        return "AGENT_WON_ELECTION"
    elif event_type == igs.AGENT_LOST_ELECTION:
        return "AGENT_LOST_ELECTION"
    else:
        return "UNKNOWN"

def signal_handler(signal_received, frame):
    global is_interrupted
    print("\n", signal.strsignal(signal_received), sep="")
    is_interrupted = True


def on_agent_event_callback(event, uuid, name, event_data, my_data):
    try:
        agent_object = my_data
        assert isinstance(agent_object, Chat)
        # add code here if needed
    except:
        print(traceback.format_exc())


def on_freeze_callback(is_frozen, my_data):
    try:
        agent_object = my_data
        assert isinstance(agent_object, Chat)
        # add code here if needed
    except:
        print(traceback.format_exc())


# inputs
def lastChatMessage_input_callback(iop_type, name, value_type, value, my_data):
    try:
        agent_object = my_data
        assert isinstance(agent_object, Chat)
        print(f"Input {name} of type {value_type} has been written with value '{value}' and user data '{my_data}'")
        agent_object.lastChatMessageI = value
        # interprétation du message
        analyse(value)
    except:
        print(traceback.format_exc())

def analyse(texte):
    print(texte)
    # on regarde si le message vient de nous
    pattern_gerant = re.compile(r'(gerant:)')
    # on regarde si l'utilisateur veut creer un musee
    pattern_creation = re.compile(r'(creer)|(crée)|(ouvre)|(ouvrir)')
    # ou bien ajouter un tableau au musee
    pattern_ajout = re.compile(r'(ajout)')
    # ou bien supprimer un tableau au musee
    pattern_suppression = re.compile(r'(enleve)|(enlève)|(supprime)')
    # ou bien fermer le musee
    pattern_fermeture = re.compile(r'(ferme)') 
    if pattern_gerant.search(texte):
        # si c'est un message de nous meme on ne fait rien
        pass
    elif pattern_creation.search(texte):
        creation_musee(texte)
    elif pattern_ajout.search(texte):
        pass
    elif pattern_suppression.search(texte):
        pass
    elif pattern_fermeture.search(texte):
        pass
    else:
        # on ne peut pas interpréter ce message
        igs.info("Votre message n'est pas interprétable, veuillez reformuler svp...")

def creation_musee(texte):
    # le message doit être de type: "(formulation) + (verbe d'ouverture du musée) + musée +- de + X +- tableaux +- de couleur + (couleur)"
    # on nettoie le message
    pattern = re.compile(r'(musée)|(musee)')
    fin = (pattern.search(texte)).end()
    texte = texte[fin:]
    # on recupere le numero de tableaux a mettre
    pattern = re.compile(r'[0123456789]{1,}')
    nb_tableaux = pattern.search(texte)
    nb_tableaux = nb_tableaux.group(0)
    # on recupere le theme du musée
    list_couleur = re.finditer(r'(tableau)|(couleur)',texte)
    for match in list_couleur:
        fin = match.end()
    couleur = texte[fin+1:]
    couleur = couleur.replace(" ", "")

    # on set l'output commande qui va appeler l'écriture sur l'entrée de IA_tableau
    # igs.service_call("Whiteboard", "clear", (), "")
    igs.output_set_string("commande","ouverture-"+couleur+"-"+nb_tableaux)
    # igs.service_call("IA_tableau", "creation_tableau_callback", couleur, "")


if __name__ == "__main__":

    # catch SIGINT handler before starting agent
    signal.signal(signal.SIGINT, signal_handler)
    interactive_loop = False

    try:
        opts, args = getopt.getopt(sys.argv[1:], short_flag, long_flag)
        # print("opts: ",opts)
        # print("args: ",args)
    except getopt.GetoptError as err:
        igs.error(err)
        sys.exit(2)
    for a in range(len(args)):
        if args[a] == "-p" or args[a] == "--port":
            port = int(args[a+1])
            # print("port: ",args[a+1])
        elif args[a] == "-d" or args[a] == "--device":
            device = args[a+1]
            # print("device: ",args[a+1])
        else:
            pass
    # print("argument: ",sys.argv)

    igs.agent_set_name(agent_name)
    igs.definition_set_version("1.0")
    igs.log_set_console(verbose)
    igs.log_set_file(True, None)
    igs.log_set_stream(verbose)
    igs.set_command_line(sys.executable + " " + " ".join(sys.argv))

    if device is None:
        # we have no device to start with: try to find one
        list_devices = igs.net_devices_list()
        list_addresses = igs.net_addresses_list()
        if len(list_devices) == 1:
            device = list_devices[0]
            igs.info("using %s as default network device (this is the only one available)" % str(device))
        elif len(list_devices) == 2 and (list_addresses[0] == "127.0.0.1" or list_addresses[1] == "127.0.0.1"):
            if list_addresses[0] == "127.0.0.1":
                device = list_devices[1]
            else:
                device = list_devices[0]
            print("using %s as de fault network device (this is the only one available that is not the loopback)" % str(device))
        else:
            if len(list_devices) == 0:
                igs.error("No network device found: aborting.")
            else:
                igs.error("No network device passed as command line parameter and several are available.")
                print("Please use one of these network devices:")
                for device in list_devices:
                    print("	", device)
                print_usage()
            exit(1)

    agent = Chat()

    igs.observe_agent_events(on_agent_event_callback, agent)
    igs.observe_freeze(on_freeze_callback, agent)

    igs.input_create("lastChatMessage", igs.STRING_T, None)

    igs.output_create("commande", igs.STRING_T, None)

    igs.observe_input("lastChatMessage", lastChatMessage_input_callback, agent)

    igs.start_with_device(device, port)
    # catch SIGINT handler after starting agent
    signal.signal(signal.SIGINT, signal_handler)

    if interactive_loop:
        print_usage_help()
        while True:
            command = input()
            if command == "/quit":
                break
            elif command == "/help":
                print_usage_help()
    else:
        while (not is_interrupted) and igs.is_started():
            time.sleep(2)

    if igs.is_started():
        igs.stop()
