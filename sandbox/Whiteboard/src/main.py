#!/usr/bin/env -P /usr/bin:/usr/local/bin python3 -B
# coding: utf-8

#
#  main.py
#  Whiteboard
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

from Whiteboard import *

port = 5670
agent_name = "Whiteboard"
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
        assert isinstance(agent_object, Whiteboard)
        # add code here if needed
    except:
        print(traceback.format_exc())


def on_freeze_callback(is_frozen, my_data):
    try:
        agent_object = my_data
        assert isinstance(agent_object, Whiteboard)
        # add code here if needed
    except:
        print(traceback.format_exc())


# inputs
def title_input_callback(iop_type, name, value_type, value, my_data):
    try:
        agent_object = my_data
        assert isinstance(agent_object, Whiteboard)
        agent_object.titleI = value
        # add code here if needed
    except:
        print(traceback.format_exc())

def backgroundColor_input_callback(iop_type, name, value_type, value, my_data):
    try:
        agent_object = my_data
        assert isinstance(agent_object, Whiteboard)
        agent_object.backgroundColorI = value
        # add code here if needed
    except:
        print(traceback.format_exc())

def chatMessage_input_callback(iop_type, name, value_type, value, my_data):
    try:
        agent_object = my_data
        assert isinstance(agent_object, Whiteboard)
        agent_object.chatMessageI = value
        # add code here if needed
    except:
        print(traceback.format_exc())

def clear_input_callback(iop_type, name, value_type, value, my_data):
    try:
        agent_object = my_data
        assert isinstance(agent_object, Whiteboard)
        # add code here if needed
    except:
        print(traceback.format_exc())

def ui_command_input_callback(iop_type, name, value_type, value, my_data):
    try:
        agent_object = my_data
        assert isinstance(agent_object, Whiteboard)
        agent_object.ui_commandI = value
        # add code here if needed
    except:
        print(traceback.format_exc())

# services
def chat_callback(sender_agent_name, sender_agent_uuid, service_name, tuple_args, token, my_data):
    try:
        agent_object = my_data
        assert isinstance(agent_object, Whiteboard)
        message = tuple_args[0]
        agent_object.chat(sender_agent_name, sender_agent_uuid, message)
    except:
        print(traceback.format_exc())


def snapshot_callback(sender_agent_name, sender_agent_uuid, service_name, tuple_args, token, my_data):
    try:
        agent_object = my_data
        assert isinstance(agent_object, Whiteboard)
        agent_object.snapshot(sender_agent_name, sender_agent_uuid)
    except:
        print(traceback.format_exc())


def clear_callback(sender_agent_name, sender_agent_uuid, service_name, tuple_args, token, my_data):
    try:
        agent_object = my_data
        assert isinstance(agent_object, Whiteboard)
        agent_object.clear(sender_agent_name, sender_agent_uuid)
    except:
        print(traceback.format_exc())


def addShape_callback(sender_agent_name, sender_agent_uuid, service_name, tuple_args, token, my_data):
    try:
        agent_object = my_data
        assert isinstance(agent_object, Whiteboard)
        type = tuple_args[0]
        x = tuple_args[1]
        y = tuple_args[2]
        width = tuple_args[3]
        height = tuple_args[4]
        fill = tuple_args[5]
        stroke = tuple_args[6]
        strokeWidth = tuple_args[7]
        agent_object.addShape(sender_agent_name, sender_agent_uuid, type, x, y, width, height, fill, stroke, strokeWidth)
    except:
        print(traceback.format_exc())


def addText_callback(sender_agent_name, sender_agent_uuid, service_name, tuple_args, token, my_data):
    try:
        agent_object = my_data
        assert isinstance(agent_object, Whiteboard)
        text = tuple_args[0]
        x = tuple_args[1]
        y = tuple_args[2]
        color = tuple_args[3]
        agent_object.addText(sender_agent_name, sender_agent_uuid, text, x, y, color)
    except:
        print(traceback.format_exc())


def addImage_callback(sender_agent_name, sender_agent_uuid, service_name, tuple_args, token, my_data):
    try:
        agent_object = my_data
        assert isinstance(agent_object, Whiteboard)
        base64 = tuple_args[0]
        x = tuple_args[1]
        y = tuple_args[2]
        width = tuple_args[3]
        height = tuple_args[4]
        agent_object.addImage(sender_agent_name, sender_agent_uuid, base64, x, y, width, height)
    except:
        print(traceback.format_exc())


def addImageFromUrl_callback(sender_agent_name, sender_agent_uuid, service_name, tuple_args, token, my_data):
    try:
        agent_object = my_data
        assert isinstance(agent_object, Whiteboard)
        url = tuple_args[0]
        x = tuple_args[1]
        y = tuple_args[2]
        agent_object.addImageFromUrl(sender_agent_name, sender_agent_uuid, url, x, y)
    except:
        print(traceback.format_exc())


def remove_callback(sender_agent_name, sender_agent_uuid, service_name, tuple_args, token, my_data):
    try:
        agent_object = my_data
        assert isinstance(agent_object, Whiteboard)
        elementId = tuple_args[0]
        agent_object.remove(sender_agent_name, sender_agent_uuid, elementId)
    except:
        print(traceback.format_exc())


def translate_callback(sender_agent_name, sender_agent_uuid, service_name, tuple_args, token, my_data):
    try:
        agent_object = my_data
        assert isinstance(agent_object, Whiteboard)
        elementId = tuple_args[0]
        dx = tuple_args[1]
        dy = tuple_args[2]
        agent_object.translate(sender_agent_name, sender_agent_uuid, elementId, dx, dy)
    except:
        print(traceback.format_exc())


def moveTo_callback(sender_agent_name, sender_agent_uuid, service_name, tuple_args, token, my_data):
    try:
        agent_object = my_data
        assert isinstance(agent_object, Whiteboard)
        elementId = tuple_args[0]
        x = tuple_args[1]
        y = tuple_args[2]
        agent_object.moveTo(sender_agent_name, sender_agent_uuid, elementId, x, y)
    except:
        print(traceback.format_exc())


def setStringProperty_callback(sender_agent_name, sender_agent_uuid, service_name, tuple_args, token, my_data):
    try:
        agent_object = my_data
        assert isinstance(agent_object, Whiteboard)
        elementId = tuple_args[0]
        property = tuple_args[1]
        value = tuple_args[2]
        agent_object.setStringProperty(sender_agent_name, sender_agent_uuid, elementId, property, value)
    except:
        print(traceback.format_exc())


def setDoubleProperty_callback(sender_agent_name, sender_agent_uuid, service_name, tuple_args, token, my_data):
    try:
        agent_object = my_data
        assert isinstance(agent_object, Whiteboard)
        elementId = tuple_args[0]
        property = tuple_args[1]
        value = tuple_args[2]
        agent_object.setDoubleProperty(sender_agent_name, sender_agent_uuid, elementId, property, value)
    except:
        print(traceback.format_exc())


def getElementIds_callback(sender_agent_name, sender_agent_uuid, service_name, tuple_args, token, my_data):
    try:
        agent_object = my_data
        assert isinstance(agent_object, Whiteboard)
        agent_object.getElementIds(sender_agent_name, sender_agent_uuid)
    except:
        print(traceback.format_exc())


def getElements_callback(sender_agent_name, sender_agent_uuid, service_name, tuple_args, token, my_data):
    try:
        agent_object = my_data
        assert isinstance(agent_object, Whiteboard)
        agent_object.getElements(sender_agent_name, sender_agent_uuid)
    except:
        print(traceback.format_exc())


if __name__ == "__main__":

    # catch SIGINT handler before starting agent
    signal.signal(signal.SIGINT, signal_handler)
    interactive_loop = False

    try:
        opts, args = getopt.getopt(sys.argv[1:], short_flag, long_flag)
    except getopt.GetoptError as err:
        igs.error(err)
        sys.exit(2)
    for o, a in opts:
        if o == "-h" or o == "--help":
            print_usage()
            exit(0)
        elif o == "-v" or o == "--verbose":
            verbose = True
        elif o == "-i" or o == "--interactive_loop":
            interactive_loop = True
        elif o == "-p" or o == "--port":
            port = int(a)
        elif o == "-d" or o == "--device":
            device = a
        elif o == "-n" or o == "--name":
            agent_name = a
        else:
            assert False, "unhandled option"

    igs.agent_set_name(agent_name)
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

    agent = Whiteboard()

    igs.observe_agent_events(on_agent_event_callback, agent)
    igs.observe_freeze(on_freeze_callback, agent)

    igs.input_create("title", igs.STRING_T, None)
    igs.input_create("backgroundColor", igs.STRING_T, None)
    igs.input_create("chatMessage", igs.STRING_T, None)
    igs.input_create("clear", igs.IMPULSION_T, None)
    igs.input_create("ui_command", igs.STRING_T, None)

    igs.output_create("lastChatMessage", igs.STRING_T, None)
    igs.output_create("lastAction", igs.STRING_T, None)
    igs.output_create("ui_error", igs.STRING_T, None)

    igs.observe_input("title", title_input_callback, agent)
    igs.observe_input("backgroundColor", backgroundColor_input_callback, agent)
    igs.observe_input("chatMessage", chatMessage_input_callback, agent)
    igs.observe_input("clear", clear_input_callback, agent)
    igs.observe_input("ui_command", ui_command_input_callback, agent)

    igs.service_init("chat", chat_callback, agent)
    igs.service_arg_add("chat", "message", igs.STRING_T)
    igs.service_init("snapshot", snapshot_callback, agent)
    igs.service_init("clear", clear_callback, agent)
    igs.service_init("addShape", addShape_callback, agent)
    igs.service_arg_add("addShape", "type", igs.STRING_T)
    igs.service_arg_add("addShape", "x", igs.DOUBLE_T)
    igs.service_arg_add("addShape", "y", igs.DOUBLE_T)
    igs.service_arg_add("addShape", "width", igs.DOUBLE_T)
    igs.service_arg_add("addShape", "height", igs.DOUBLE_T)
    igs.service_arg_add("addShape", "fill", igs.STRING_T)
    igs.service_arg_add("addShape", "stroke", igs.STRING_T)
    igs.service_arg_add("addShape", "strokeWidth", igs.DOUBLE_T)
    igs.service_init("addText", addText_callback, agent)
    igs.service_arg_add("addText", "text", igs.STRING_T)
    igs.service_arg_add("addText", "x", igs.DOUBLE_T)
    igs.service_arg_add("addText", "y", igs.DOUBLE_T)
    igs.service_arg_add("addText", "color", igs.STRING_T)
    igs.service_init("addImage", addImage_callback, agent)
    igs.service_arg_add("addImage", "base64", igs.DATA_T)
    igs.service_arg_add("addImage", "x", igs.DOUBLE_T)
    igs.service_arg_add("addImage", "y", igs.DOUBLE_T)
    igs.service_arg_add("addImage", "width", igs.DOUBLE_T)
    igs.service_arg_add("addImage", "height", igs.DOUBLE_T)
    igs.service_init("addImageFromUrl", addImageFromUrl_callback, agent)
    igs.service_arg_add("addImageFromUrl", "url", igs.STRING_T)
    igs.service_arg_add("addImageFromUrl", "x", igs.DOUBLE_T)
    igs.service_arg_add("addImageFromUrl", "y", igs.DOUBLE_T)
    igs.service_init("remove", remove_callback, agent)
    igs.service_arg_add("remove", "elementId", igs.INTEGER_T)
    igs.service_init("translate", translate_callback, agent)
    igs.service_arg_add("translate", "elementId", igs.INTEGER_T)
    igs.service_arg_add("translate", "dx", igs.DOUBLE_T)
    igs.service_arg_add("translate", "dy", igs.DOUBLE_T)
    igs.service_init("moveTo", moveTo_callback, agent)
    igs.service_arg_add("moveTo", "elementId", igs.INTEGER_T)
    igs.service_arg_add("moveTo", "x", igs.DOUBLE_T)
    igs.service_arg_add("moveTo", "y", igs.DOUBLE_T)
    igs.service_init("setStringProperty", setStringProperty_callback, agent)
    igs.service_arg_add("setStringProperty", "elementId", igs.INTEGER_T)
    igs.service_arg_add("setStringProperty", "property", igs.STRING_T)
    igs.service_arg_add("setStringProperty", "value", igs.STRING_T)
    igs.service_init("setDoubleProperty", setDoubleProperty_callback, agent)
    igs.service_arg_add("setDoubleProperty", "elementId", igs.INTEGER_T)
    igs.service_arg_add("setDoubleProperty", "property", igs.STRING_T)
    igs.service_arg_add("setDoubleProperty", "value", igs.DOUBLE_T)
    igs.service_init("getElementIds", getElementIds_callback, agent)
    igs.service_init("getElements", getElements_callback, agent)

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
