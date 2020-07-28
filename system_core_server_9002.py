import json

from websocket_server import WebsocketServer
# Called for every client connecting (after handshake)
def new_client(client, server):
    print("New client connected and was given id %d" % client['id'])


# Called for every client disconnecting
def client_left(client, server):
    print("Client(%d) disconnected" % client['id'])


# Called when a client sends a message
def message_received(client, server, message):
    if len(message) > 200:
        message = message[:200]+'..'
    # print("MESSAGE: %s" % json.loads(message))
    message = str(json.loads(message))
    if chr(39) in message:
        message = message.replace(chr(39), chr(34))
    print("MESSAGE: %s" % message)
    server.send_message_to_all("%s" % message)


PORT=9002
server = WebsocketServer(PORT)
server.set_fn_new_client(new_client)
server.set_fn_client_left(client_left)
server.set_fn_message_received(message_received)
server.run_forever()