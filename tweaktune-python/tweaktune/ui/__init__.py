import queue
import threading
from nicegui import ui

def render_graph(graph):
    graph_str = """---
config:
  layout: elk
  look: handDrawn
  theme: dark  
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

        if ix == len(graph) - 1:
            graph_str += f'  STEP{ix}["**{step_func}**<br/>{step_args}"]'
        else:
            graph_str += f'  STEP{ix}["**{step_func}**<br/>{step_args}"] -->'

    return graph_str

def run_ui(builder, graph, config, host: str="0.0.0.0", port: int=8080):

    bus = queue.Queue()

    ui.add_head_html("""
    <style>
        /*.q-btn {
            background-color: #555 !important;
        }*/
                     
        .q-drawer {
            width: 55px !important;
            transition: width 0.3s !important;
            overflow-x: hidden !important;
        }
                     
        .q-drawer:hover {
            width: 220px !important;
        }
                     
        .q-drawer .q-drawer__content {
            padding: 0.5em !important;
            overflow-x: hidden !important;
            white-space: nowrap !important;
            text-overflow: ellipsis !important;
        }
                     
        .q-drawer::-webkit-scrollbar,
        .q-drawer__content::-webkit-scrollbar {
            display: none !important;
        }
    </style>
    """)

    def run_builder_thread(bus):
        builder.run(bus)

    def run_builder():
        ui.notify('Running builder !')
        spinner.visible = True
        threading.Thread(target=lambda: run_builder_thread(bus), daemon=False).start()

    #with ui.footer().classes('bg-dark'), ui.column().classes('w-full max-w-3xl mx-auto my-6'):

    with ui.column().classes('h-10'):
        ui.space()

    with ui.tabs().classes('w-full') as tabs:
        one = ui.tab('Graph')
        two = ui.tab('Logs')
    with ui.tab_panels(tabs, value=two).classes('w-full'):
        with ui.tab_panel(one).classes('flex flex-col items-center justify-center'):
            ui.mermaid(render_graph(graph))
        with ui.tab_panel(two):
            log = ui.log(max_lines=10).classes('w-full h-200')

    with ui.page_sticky(x_offset=18, y_offset=18, position="bottom"):
        with ui.row().classes('items-center w-full'):
            ui.button(icon='play_arrow', on_click=run_builder).props('fab color=green')
            ui.button(icon='stop', on_click=run_builder).props('fab color=red')
            spinner = ui.spinner('audio', size='lg', color='green')
            spinner.visible = False



    def check_bus():
        while not bus.empty():
            message = bus.get()
            if message is None:
                break
            log.push(message)
            #ui.notify(message)

    #ui.dark_mode().enable()
    #with ui.dialog() as dialog_graph, ui.card():
    #    ui.button('Close', on_click=dialog_graph.close)

    #ui.button('Graph', on_click=dialog_graph.open)

#    with ui.dialog() as dialog_config, ui.card():
#        ui.button('Close', on_click=dialog_config.close)
#        ui.mermaid(render_graph(config))

#    ui.button('Graph', on_click=dialog_config.open)

    #with ui.row().classes('text-4xl'):
    #    ui.icon('home')
    #    ui.icon('o_home')
    #    ui.icon('r_home')
    #    ui.icon('sym_o_home')
    #    ui.icon('sym_r_home')

    #with ui.header(elevated=True).style('background-color: #555').classes('items-center justify-between'):
    #    ui.label('HEADER')
    #    #ui.button(on_click=lambda: right_drawer.toggle(), icon='menu').props('flat color=white')
    #with ui.left_drawer(fixed=False, value=True).props('bordered') as right_drawer:
    #    ui.icon('home')
    #    ui.label('Right Drawer')
    #    #ui.mermaid(render_graph(graph))
    #with ui.footer().style('background-color: #555'):
    #    ui.label('FOOTER')




    ui.timer(0.2, check_bus, once=False)

    ui.run(host="0.0.0.0", port=8080, dark=True, title='TweakTune UI', favicon='https://tweaktune.org/favicon.ico')