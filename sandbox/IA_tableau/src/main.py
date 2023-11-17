#!/usr/bin/env -P /usr/bin:/usr/local/bin python3 -B
# coding: utf-8

#
#  main.py
#  IA_tableau version 1.0
#  Created by Ingenuity i/o on 2023/11/16
#
# ia chargé de générer des images
#

import signal
import getopt
import time
from pathlib import Path
import traceback
import sys

from matplotlib.pyplot import imshow
import generateur
import numpy as np
import base64
from io import BytesIO
from PIL import Image

from IA_tableau import *

port = 15670
agent_name = "IA_tableau"
device = None
verbose = False
is_interrupted = False

short_flag = "hvip:d:n:"
long_flag = ["help", "verbose", "interactive_loop", "port=", "device=", "name="]

ingescape_path = Path("~/Documents/Ingescape").expanduser()

gallerie_tableaux = []

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
        assert isinstance(agent_object, IATableau)
        # add code here if needed
    except:
        print(traceback.format_exc())


def on_freeze_callback(is_frozen, my_data):
    try:
        agent_object = my_data
        assert isinstance(agent_object, IATableau)
        # add code here if needed
    except:
        print(traceback.format_exc())


# inputs
def action_input_callback(iop_type, name, value_type, value, my_data):
    try:
        print(f"Input {name} of type {value_type} has been written with value '{value}' and user data '{my_data}'")
        agent_object = my_data
        assert isinstance(agent_object, IATableau)
        agent_object.actionI = value
        # on décortique la commande recu
        informations = value.split('-')
        type_action = informations[0]
        nb_tableaux = int(informations[2])
        couleur = informations[1]
        # on appel la fonction correspondante 
        if type_action == "ouverture":
            if len(gallerie_tableaux) == 0:
                if nb_tableaux==0:
                    # aucun tableaux ajouté au musée crée, on le spécifie dans le chat
                    igs.service_call("Whiteboard", "chat", "gerant: Ouvrir un musée vide...Quelle drôle d'idée...", "")
                elif nb_tableaux>12:
                    # aucun tableaux ajouté au musée crée, on le spécifie dans le chat
                    igs.service_call("Whiteboard", "chat", "gerant: Oh...Notre musée est trop petit pour accueillir tant d'oeuvres ;-; Belle ambition nonobstant...", "")
                else:
                    for i in range(nb_tableaux):
                        if i!=0:
                            if x>=550.0:
                                y += 150.0
                                x = 100.0
                            else:    
                                x += 150.0
                        else:
                            y = 50.0
                            x = 100.0
                        tableau = creation_tableau(couleur,x,y,250,250)
                        gallerie_tableaux.append(tableau)

        elif type_action == "ajout":
            pass
        elif type_action == "fermeture":
            pass
        elif type_action == "suppression":
            pass
        else:
            # action non reconnue
            pass
    except:
        print(traceback.format_exc())

def creation_tableau(couleur,x, y, width, height):
    if len(couleur)<4:
        image = generateur.generation_image_sans_couleur()
    else:
        image = generateur.generation_image(couleur)
    image_b64 = image_to_base64(image)
    # base64, x, y, width, height
    parametre_image = (image_b64, x, y, width, height)
    igs.service_call("Whiteboard", "addImage", parametre_image, "")
    print("AFFICHAGE IMAGE")
    return parametre_image


def image_to_base64(image_matrix):
    # Convert the image matrix to a uint8 format (assuming it is in the range [0, 1])
    image_matrix =  Image.fromarray((image_matrix * 255).astype(np.uint8))
    # Create a BytesIO object to store the image in memory
    image_buffer = BytesIO()
    # Save the PIL Image to the BytesIO object in PNG format
    image_matrix.save(image_buffer, format='PNG')
    imshow(image_matrix)
    # Encode the image in base64
    base64_encoded = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
    return base64_encoded

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
    igs.definition_set_description("""ia chargé de générer des images""")
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

    agent = IATableau()

    igs.observe_agent_events(on_agent_event_callback, agent)
    igs.observe_freeze(on_freeze_callback, agent)

    igs.input_create("action", igs.STRING_T, None)

    igs.observe_input("action", action_input_callback, agent)

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

