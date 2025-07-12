import queue
import threading
from nicegui import ui

def render_graph(graph):
    graph_str = """---
config:
  look: handDrawn
  theme: neutral
  darkMode: true
---
graph TD;\n"""
    for ix,step in enumerate(graph):
        step_name = step.name
        step_func = step.func
        step_args = step.args
        if "self" in step_args:
            del step_args['self']

        if "name" in step_args:
            del step_args['name']

        print("STEP ARGS", step_args)

        if ix == len(graph) - 1:
            graph_str += f'  STEP{ix}["**{step_func}**<br/>{step_args}"]'
        else:
            graph_str += f'  STEP{ix}["**{step_func}**<br/>{step_args}"] -->'

    return graph_str

def run_ui(builder, graph, config, host: str="0.0.0.0", port: int=8080):

    bus = queue.Queue()

    log = ui.log(max_lines=10).classes('w-full h-200')
    def check_bus():
        while not bus.empty():
            message = bus.get()
            if message is None:
                break
            log.push(message)
            #ui.notify(message)

#    with ui.dialog() as dialog_graph, ui.card():
#        ui.button('Close', on_click=dialog_graph.close)
#        ui.mermaid(render_graph(graph))

#    ui.button('Graph', on_click=dialog_graph.open)

    with ui.dialog() as dialog_config, ui.card():
        ui.button('Close', on_click=dialog_config.close)
        ui.mermaid(render_graph(config))

    ui.button('Graph', on_click=dialog_config.open)


    with ui.header(elevated=True).style('background-color: #3874c8').classes('items-center justify-between'):
        ui.label('HEADER')
        ui.button(on_click=lambda: right_drawer.toggle(), icon='menu').props('flat color=white')
    with ui.right_drawer(fixed=False).props('bordered') as right_drawer:
        ui.label('Right Drawer')
        #ui.mermaid(render_graph(graph))
    with ui.footer().style('background-color: #3874c8'):
        ui.label('FOOTER')



    def run_builder_thread(bus):
        builder.run(bus)

    def run_builder():
        ui.notify('Running builder !')
        threading.Thread(target=lambda: run_builder_thread(bus), daemon=False).start()

    ui.button('START', on_click=run_builder)
    ui.timer(0.2, check_bus, once=False)

    ui.run(host="0.0.0.0", port=8080)