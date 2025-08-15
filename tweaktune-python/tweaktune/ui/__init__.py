import queue
import json
import threading
import inspect
from nicegui import ui
from tweaktune import Graph
from collections import deque

def renger_graph_section(graph, section: str) -> str:

    callback_str = ""
    section_str = f'  subgraph {section.upper()} [**{section}**]\n    direction TB\n  '
    for ix, config_step in enumerate(getattr(graph.config, section)):
        step_name = config_step.name
        step_func = config_step.func
        step_args = config_step.args
        if "self" in step_args:
            del step_args['self']

        name = step_args.get("name") or step_args.get("workers") or ""

        if ix == len(getattr(graph.config, section)) - 1:
            section_str += f'  {section.upper()}{ix}["**{step_func}**<br/>{name}"]\n'
        else:
            section_str += f'  {section.upper()}{ix}["**{step_func}**<br/>{name}"] -->'


        callback_str += f'\n  click {section.upper()}{ix} call emitEvent("mermaid_click", "{section.upper()}@{ix}")'

    section_str += f'  end\n  {section.upper()} --> START\n'
    return section_str, callback_str

def render_graph(graph: Graph) -> str:
    graph_str = """---
config:
  layout: elk
  look: handDrawn
  theme: dark  
  securityLevel: loose
---
graph TD;\n"""
    graph_str += f'  START[**{graph.start.func}**<br/>workers: {graph.config.workers}]\n'
    callback_str = ""
    if graph.config.llms:
        gr,cb = renger_graph_section(graph, 'llms')
        graph_str += gr
        callback_str += cb
    if graph.config.datasets:
        gr, cb = renger_graph_section(graph, 'datasets')
        graph_str += gr
        callback_str += cb
    if graph.config.templates:
        gr, cb = renger_graph_section(graph, 'templates')
        graph_str += gr
        callback_str += cb

    callback_str += f'\n  click START call emitEvent("mermaid_click", "START@0")'
    graph_str += "\n\n"

    for ix,step in enumerate(graph.steps):
        step_name = step.name
        step_func = step.func
        step_args = {k:str(v) for k,v in step.args.items()}.copy()
        if "self" in step_args:
            del step_args['self']

        if "name" in step_args:
            del step_args["name"]

        if ix == 0:
            graph_str += f'  START[**{graph.start.func}**<br/>workers: {graph.config.workers}] --> STEP{ix}["**{step_func}**<br/>{step_args}"] -->'
        elif ix == len(graph.steps) - 1:
            graph_str += f'  STEP{ix}["**{step_func}**<br/>{step_args}"]'
        else:
            graph_str += f'  STEP{ix}["**{step_func}**<br/>{step_args}"] -->'

        callback_str += f'\n  click STEP{ix} call emitEvent("mermaid_click", "STEP@{ix}")'

    graph_str = f"{graph_str}{callback_str}"
    return graph_str

def run_ui(builder, graph, host: str="0.0.0.0", port: int=8080):

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
                     
        .mermaid-center svg{
            display: block !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }
    </style>
    """)

    def run_builder_thread(bus):
        builder.run(bus)

    def run_builder():
        ui.notify('Starting builder ...')
        progress.visible = True
        #tab_panels.value = two
        threading.Thread(target=lambda: run_builder_thread(bus), daemon=False).start()

    def stop_builder():
        ui.notify('Stopping builder !')
        progress.visible = False
        builder.stop()
    
    dialog_data = ui.dialog()
    data_file = {"filename": ""}
    
    with ui.dialog() as dialog_graph, ui.card().style('width:auto; max-width: none;'): 
        graph_header = ui.markdown("")
        ui.separator()
        graph_code = ui.code('', language='json')
        graph_py = ui.code('', language='python')
        graph_template = ui.code('', language='jinja')
        data_range = ui.range(min=0, max=100, value={'min': 30, 'max': 70}).props('label-always')

        def run_data_range():
            file_data = []
            with open(data_file["filename"], 'r') as f:
                for i, line in enumerate(f):
                    if i < data_range.value['min']:
                        continue
                    if i > data_range.value['max']:
                        break
                    file_data.append(json.loads(line))
                #last_lines = deque(f, maxlen=10)
            json_data = {'content': {'json': file_data}}
            dialog_data.clear()
            with dialog_data:
                data_editor = ui.json_editor(json_data)
            dialog_data.open()

        data_button = ui.button('SHOW DATA RANGE', on_click=run_data_range)

    def hide_dialog_graph_elements():
        graph_template.visible = False
        graph_code.visible = False
        graph_template.visible = False
        graph_py.visible = False
        data_button.visible = False
        data_range.visible = False
            
        #ui.button('Close', on_click=dialog_mermaid.close)

    def run_mermaid_dialog(node):
        node = node.split('@')
        if node[0] in ['LLMS', 'DATASETS', 'TEMPLATES']:
            node_ix = int(node[1])
            config_step = getattr(graph.config, node[0].lower())[node_ix]
            func = config_step.func
            graph_header.content = f"**`{func}`**"
            if func == 'with_template':
                hide_dialog_graph_elements()
                graph_header.content += f" **(\"{config_step.args.get('name')}\")**"
                graph_template.visible = True
                graph_template.content = config_step.args.get("template")
            else:
                hide_dialog_graph_elements()
                graph_template.content = ""
                graph_code.visible = True
                graph_code.content = json.dumps(config_step.args, indent=2)

        elif node[0] == 'STEP':
            node_ix = int(node[1])
            step = graph.steps[node_ix]
            func = step.func
            if func in ['map', 'add_column', 'mutate']:
                hide_dialog_graph_elements()
                py_func = step.args.get("func")
                if inspect.isfunction(py_func):
                    graph_py.content = inspect.getsource(py_func)
                    graph_py.visible = True

                graph_header.content = f"**`{func}`**"
                if "output" in step.args:
                    graph_header.content += f" **(\"{step.args.get('output')}\")**"

            elif func in ['write_jsonl', 'write_csv', 'write_parquet']:
                hide_dialog_graph_elements()
                graph_header.content = f"**`{func}`**"
                graph_code.content = json.dumps(step.args, indent=2)
                graph_code.visible = True
                data_button.visible = True
                data_range.visible = True
                
                with open(step.args.get('path'), 'r') as f:
                    num_lines = sum(1 for _ in f)

                data_range.props(f'max={num_lines}')
                data_range.value = {'min': num_lines-20, 'max': num_lines}
                data_file["filename"]=step.args.get('path')
            else:
                hide_dialog_graph_elements()
                graph_header.content = f"**`{func}`**"
                graph_code.content = json.dumps(step.args, indent=2)
                graph_code.visible = True
        elif node[0] == 'START':
            hide_dialog_graph_elements()
            graph_header.content = f"**`{graph.start.func}`**"
            graph_code.content = json.dumps(graph.start.args, indent=2)
            graph_code.visible = True
        
        #mermaid.content = f"**TEST {node}**"
        dialog_graph.open()
        #ui.notify(f'Node {node} clicked !')

    #ui.button('Graph', on_click=dialog_graph.open)

    ui.on('mermaid_click', lambda e: run_mermaid_dialog(e.args))

    #with ui.footer().classes('bg-dark'), ui.column().classes('w-full max-w-3xl mx-auto my-6'):

    with ui.tabs().classes('w-full') as tabs:
        one = ui.tab('Graph')
        two = ui.tab('Logs')
    with ui.tab_panels(tabs, value=one).classes('w-full').style("height: 80vh;") as tab_panels:
        with ui.tab_panel(one).classes('flex flex-col items-center justify-center'):
            with ui.element('div').style('width: 80vw; height: 60vh; display: flex !important; justify-content: center !important; align-items: center !important;'):
                ui.mermaid(render_graph(graph), config={'securityLevel': 'loose'} ).classes("mermaid-center").style('width: 70vw; height: 50vh; display: block;')
        with ui.tab_panel(two):
            log = ui.log(max_lines=10).classes('w-full h-200')

    with ui.page_sticky(x_offset=18, y_offset=18, position="bottom"):
        with ui.row().classes('items-center w-full'):
            ui.button(icon='play_arrow', on_click=run_builder).props('fab color=green')
            ui.button(icon='stop', on_click=stop_builder).props('fab color=red')
            progress = ui.circular_progress()
            progress.visible = False

        
    def check_bus():
        while not bus.empty():
            message = bus.get()
            if message is None:
                break
            m = json.loads(message)
            if m['event_type'] == 'log':
                log.push(m['data'])
            elif m['event_type'] == 'progress':
                data = m['data']
                progress.value = data['index']+1
                progress.props(f"max={data['total']}")
            elif m['event_type'] == 'finished':
                ui.notify(m['data']['message'])
                data = m['data']
                progress.props(f"color=blue")
                progress.props(':thickness=1')

            #ui.notify(message)

    #ui.dark_mode().enable()

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
