import json
import torch
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from flask import Flask, render_template, request, jsonify, session # Removida duplicata de import
import os
import random
import re
import datetime
from produto_utils import identificar_produto_por_texto # Removida duplicata de import
from produto_utils import classify_text # Importado explicitamente se for de produto_utils
# Se classify_text não estiver em produto_utils, defina-o ou importe de onde estiver

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
INTENTS_FILE = "intents.json"
USERS_FILE = "users.json"
ORDERS_FILE = "orders.json"
MENU_FILE = "menu.json"

# CORREÇÃO: Mover a definição da função remover_acentos para antes do seu primeiro uso
# Existem duas definições de remover_acentos no código original. Usarei a segunda que parece mais completa com unicodedata.
def remover_acentos(texto):
    """Remove acentos e caracteres especiais de uma string."""
    import unicodedata
    return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')

# Carregar intents
with open(INTENTS_FILE, encoding="utf-8") as f:
    intents_data = json.load(f)
    intents = intents_data["intents"]
    fallback = intents_data["fallback"]

# CORREÇÃO: Mover a definição da função load_menu para antes do seu primeiro uso
def load_menu():
    """Carrega os dados do menu a partir do arquivo JSON."""
    try:
        with open(MENU_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{MENU_FILE}' não encontrado. Certifique-se de que ele está na mesma pasta do 'app.py'.")
        return [] # Retorna uma lista vazia para evitar erros
    except json.JSONDecodeError:
        print(f"Erro: Arquivo '{MENU_FILE}' contém JSON inválido.")
        return [] # Retorna uma lista vazia para evitar erros

# Carregar o menu uma vez na inicialização do aplicativo
menu_data = load_menu()

# Criar um dicionário de preços para fácil acesso
menu_prices = {remover_acentos(item["item"].lower()): item["price"] for item in menu_data if "item" in item and "price" in item}


x_texts, y_labels = [], []
for intent_item in intents: # Renomeado para evitar conflito com a variável 'intent' mais abaixo
    for pattern in intent_item["patterns"]:
        x_texts.append(pattern)
        y_labels.append(intent_item["tag"])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_labels)

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
bert = BertModel.from_pretrained(MODEL_NAME)

class BertVectorizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, texts):
        vectors = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
            with torch.no_grad():
                outputs = bert(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            vectors.append(cls_embedding)
        return vectors

pipeline = Pipeline([
    ("bert", BertVectorizer()),
    ("clf", LogisticRegression(max_iter=1000))
])

pipeline.fit(x_texts, y_encoded)

app = Flask(__name__)
conversations = {}
app.secret_key = 'fast_chatbot' # Chave aleatória e segura para produção


# A segunda definição de MENU_FILE e remover_acentos foram removidas pois já estavam definidas.
# A segunda definição de load_menu e menu_data também.

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/mensagem", methods=["POST"])
def mensagem_endpoint():
    body = request.get_json()
    input_usuario = body.get("mensagem")

    if not input_usuario:
        return jsonify({"resposta": "Por favor, envie uma mensagem válida."})

    # Supondo que menu_data já está carregado globalmente e corretamente
    produto_encontrado = identificar_produto_por_texto(input_usuario, menu_data)

    if produto_encontrado:
        return jsonify({
            "resposta": f"Você escolheu: {produto_encontrado['item']} — R$ {produto_encontrado['price']:.2f}",
            "produto": produto_encontrado
        })
    else:
        return jsonify({"resposta": "Desculpe, não entendi o produto. Poderia repetir?"})

# Menu dinâmico com opções
MENU_OPCOES = {
    "hambúrguer": "Hambúrguer",
    "hamburguer": "Hambúrguer",  # sem acento
    "pizza": "Pizza",
    "refrigerante": "Refrigerante",
    "promoções": "Promoções",
    "promocoes": "Promoções",
    "x-salada": "X-Salada",
    "xis salada": "X-Salada",
    "x-bacon": "X-Bacon",
    "xis bacon": "X-Bacon",
    "batata frita p": "Batata Frita (P)",
    "batata frita pequena": "Batata Frita (P)",
    "batata frita g": "Batata Frita (G)",
    "batata frita grande": "Batata Frita (G)",
    "onion rings": "Onion Rings",
    "nuggets": "Nuggets (10 unidades)",
    "suco": "Suco Natural",
    "agua": "Água Mineral",
    "salada caesar": "Salada Caesar",
    "salada mista": "Salada Mista",
    "brownie": "Brownie",
    "mousse": "Mousse de Chocolate",
    "sorvete": "Sorvete",
    "calabresa": "pizza calabresa", # CORREÇÃO: Nome deve ser consistente com menu.json para preços
    "marguerita": "pizza marguerita", # CORREÇÃO: Nome deve ser consistente com menu.json para preços
    "frango com catupiry": "pizza frango com Catupiry", # CORREÇÃO
    "portuguesa": "pizza portuguesa", # CORREÇÃO
    "pepperoni": "pizza pepperoni", # CORREÇÃO
    "tradicional": "Tradicional", # Hambúrguer
    "duplo": "Duplo", # Hambúrguer
    "veggie": "Veggie", # Hambúrguer
    "refrigerante lata": "Refrigerante (Lata)",
    "refrigerante 2l": "Refrigerante (2L)",
}

# Mapear números escritos para números digitados (1-10)
NUMEROS_MAP = {
    "uma": 1, "um": 1,
    "duas": 2, "dois": 2,
    "três": 3, "tres": 3,
    "quatro": 4,
    "cinco": 5,
    "seis": 6,
    "sete": 7,
    "oito": 8,
    "nove": 9,
    "dez": 10
}

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, encoding='utf-8') as f:
            return json.load(f)
    return []

def save_users(users):
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def find_user_by_cpf(cpf):
    for u in load_users():
        if u['cpf'] == cpf:
            return u
    return None

def load_orders():
    try:
        with open(ORDERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}

def save_orders(orders):
    with open(ORDERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(orders, f, ensure_ascii=False, indent=2)

def append_order(cpf, new_order_item): # Modificado para adicionar um item de pedido, não uma lista
    orders = load_orders()
    if cpf not in orders:
        orders[cpf] = [] # Garante que é uma lista
    
    # Verifica se o item já existe para incrementar a quantidade
    # Esta lógica deveria estar em combinar_itens_pedido ou adicionar_item_ao_pedido
    # append_order deve apenas salvar o estado atual do pedido do contexto
    # Para simplificar, vamos assumir que new_order_item é o pedido completo (lista de dicts)
    orders[cpf] = new_order_item # Salva/sobrescreve o pedido atual do usuário
    save_orders(orders)


def predict_intent(message):
    try:
        pred = pipeline.predict([message])[0]
        tag = label_encoder.inverse_transform([pred])[0]
        for intent_item_pred in intents: # Renomeado para evitar conflito
            if intent_item_pred["tag"] == tag:
                return random.choice(intent_item_pred["responses"]), tag
    except Exception as e:
        print(f"Erro ao classificar: {e}")
    return fallback, None

def parse_order(message):
    """
    Extrai itens e quantidades do pedido na frase.
    Exemplo: "quero 2 hambúrgueres e 1 pizza"
    Retorna lista de dicts: [{'item': 'Hambúrguer', 'quantidade': 2}, ...]
    """
    items = []
    msg = remover_acentos(message.lower())
    pattern = r'(\d+|uma|um|duas|dois|tres|três|quatro|cinco|seis|sete|oito|nove|dez)\s+([a-z0-9\s-]+)'
    matches = re.findall(pattern, msg)

    for quant_str, item_str in matches:
        quant = 1
        if quant_str.isdigit():
            quant = int(quant_str)
        elif quant_str in NUMEROS_MAP:
            quant = NUMEROS_MAP[quant_str]

        item_base = item_str.strip()
        item_mapped = None
        # Busca mais específica primeiro (ex: "pizza calabresa")
        for key_menu, val_menu in MENU_OPCOES.items():
            key_norm = remover_acentos(key_menu.lower()).strip()
            # Verifica se item_base é exatamente um alias em MENU_OPCOES
            if item_base == key_norm:
                item_mapped = val_menu
                break
            # Verifica se item_base contém um alias mais longo
            if key_norm in item_base: # Ex: "pizza calabresa" em "quero uma pizza calabresa"
                 # Tenta ser mais específico, se "pizza calabresa" está em MENU_OPCOES, use.
                if item_base.startswith(key_norm) or item_base.endswith(key_norm) or key_norm == item_base:
                    item_mapped = val_menu
                    break


        if item_mapped:
            items.append({"item": item_mapped, "quantidade": quant})
        else: # Fallback para itens simples se não encontrou combinação quant+item complexo
            for key_menu, val_menu in MENU_OPCOES.items():
                key_norm = remover_acentos(key_menu.lower()).strip()
                if key_norm in msg: # Se "calabresa" está na msg e em MENU_OPCOES
                    items.append({"item": val_menu, "quantidade": quant if item_str.strip() == key_norm else 1}) # Usa quant se o match foi exato
                    break # Pega o primeiro que encontrar

    if not items: # Se o regex de quantidade falhou, tenta pegar só o item
        for key_menu, val_menu in MENU_OPCOES.items():
            key_norm = remover_acentos(key_menu.lower()).strip()
            if key_norm in msg:
                items.append({"item": val_menu, "quantidade": 1})
                break
    return items


def combinar_itens_pedido(pedido):
    combinados = {}
    for entrada in pedido:
        chave = entrada["item"] # Assume que 'item' é o nome canônico de MENU_OPCOES
        if chave in combinados:
            combinados[chave] += entrada["quantidade"]
        else:
            combinados[chave] = entrada["quantidade"]
    return [{"item": k, "quantidade": v} for k, v in combinados.items()]

def remover_item_do_pedido(pedido, item_nome, quantidade=None):
    removidos = 0
    item_nome_norm = remover_acentos(item_nome.lower())
    # Mapeia o nome fornecido para o nome canônico do menu, se necessário
    item_canonico_para_remover = None
    for key_menu, val_menu in MENU_OPCOES.items():
        if remover_acentos(key_menu.lower()) == item_nome_norm:
            item_canonico_para_remover = val_menu
            break
    if not item_canonico_para_remover:
        item_canonico_para_remover = item_nome # Usa o nome original se não mapeado

    for i in range(len(pedido) - 1, -1, -1):
        # Compara o nome canônico do item no pedido com o nome canônico a ser removido
        if pedido[i]["item"] == item_canonico_para_remover:
            if quantidade is None or pedido[i]["quantidade"] <= quantidade:
                removidos_nesta_iteracao = pedido[i]["quantidade"]
                pedido.pop(i)
                removidos += removidos_nesta_iteracao
                if quantidade is not None: # Se especificou quantidade e removeu tudo ou menos
                    return removidos # Retorna o total removido
            elif pedido[i]["quantidade"] > quantidade:
                pedido[i]["quantidade"] -= quantidade
                removidos = quantidade
                return removidos # Retorna a quantidade especificada que foi removida
    return removidos


def visualizar_pedido(pedido_atual): # Renomeado para evitar conflito
    if not pedido_atual:
        return "Seu carrinho está vazio."

    resumo = "📋 *Resumo do Pedido:*\n"
    total = 0.0

    for item_dict in pedido_atual: # Renomeado para evitar conflito
        nome_item = item_dict["item"] # Este deve ser o nome canônico de MENU_OPCOES
        quantidade = item_dict["quantidade"]
        
        # A chave para menu_prices é o nome do item em menu.json normalizado
        # Precisamos encontrar o nome original do menu.json correspondente ao nome_item (de MENU_OPCOES)
        chave_preco = ""
        for item_menu_json in menu_data:
            # Compara o valor de MENU_OPCOES (nome_item) com o item em menu.json
            # Isso pode ser complexo se os nomes não forem 1 para 1 ou se MENU_OPCOES mapeia para um nome diferente do "item" em menu.json
            # Para simplificar, vamos assumir que o nome_item (valor de MENU_OPCOES) é o que está em menu.json após normalização
            if remover_acentos(item_menu_json["item"].lower()) == remover_acentos(nome_item.lower()):
                 chave_preco = remover_acentos(item_menu_json["item"].lower())
                 break
        
        if not chave_preco: # Fallback se não encontrou mapeamento direto
            chave_preco = remover_acentos(nome_item.lower())


        preco_unitario = menu_prices.get(chave_preco)

        if preco_unitario is None:
            resumo += f"- {quantidade}x {nome_item}: preço não encontrado ❌\n"
            print(f"DEBUG: Preço não encontrado para '{nome_item}' (chave usada: '{chave_preco}')")
            continue

        subtotal = preco_unitario * quantidade
        total += subtotal
        resumo += f"- {quantidade}x {nome_item} — R$ {preco_unitario:.2f} = R$ {subtotal:.2f}\n"

    resumo += f"\n💰 *Total do Pedido:* R$ {total:.2f}"
    return resumo

def formatar_pedido_para_exibir(pedido_atual, exibir_total=False): # Renomeado
    if not pedido_atual:
        return "Seu carrinho está vazio."

    texto = ""
    total = 0.0

    for item_dict in pedido_atual: # Renomeado
        nome = item_dict.get("item", "Item desconhecido")
        qtd = item_dict.get("quantidade", 1)
        texto += f"- {qtd}x {nome}\n"

        if exibir_total:
            chave_preco_format = remover_acentos(nome.lower())
            # Tentativa de encontrar a chave correta em menu_prices
            # Esta lógica precisa ser robusta, similar a `visualizar_pedido`
            preco_encontrado = False
            for item_menu_json_fmt in menu_data:
                 if remover_acentos(item_menu_json_fmt["item"].lower()) == chave_preco_format:
                      chave_preco_format = remover_acentos(item_menu_json_fmt["item"].lower())
                      preco_encontrado = True
                      break
            if not preco_encontrado: # Fallback
                 pass # chave_preco_format já é nome.lower() normalizado

            preco = menu_prices.get(chave_preco_format, 0)
            subtotal = preco * qtd
            total += subtotal
            texto += f"  ↳ R$ {preco:.2f} x {qtd} = R$ {subtotal:.2f}\n"

    if exibir_total:
        texto += f"\n💰 Total do pedido: R$ {total:.2f}"
    return texto

# unicodedata já importado em remover_acentos

def handle_menu_request():
    menu_str = "Nosso Menu:\n"
    # Itera sobre menu_data (lista de dicts de menu.json) para exibir nomes originais e preços
    for item_menu in menu_data:
        nome_item_menu = item_menu["item"]
        preco_item_menu = item_menu["price"]
        menu_str += f"- {nome_item_menu.capitalize()}: R$ {preco_item_menu:.2f}\n"
    return jsonify({"response": menu_str})


# CORREÇÃO: adicionar_item_ao_pedido não deve retornar jsonify.
# Ela é chamada dentro de handle_order, que já retorna jsonify.
def adicionar_item_ao_pedido(pedido_lista, item_nome, quantidade): # Renomeado pedido para pedido_lista
    """Adiciona um item ao pedido ou atualiza a quantidade se já existir.
       Modifica a lista 'pedido_lista' diretamente.
       'item_nome' deve ser o nome canônico (valor de MENU_OPCOES).
    """
    item_encontrado_flag = False
    for item_existente in pedido_lista:
        # Compara o nome canônico do item_existente com o item_nome a ser adicionado
        if item_existente["item"] == item_nome:
            item_existente["quantidade"] += quantidade
            item_encontrado_flag = True
            break
    if not item_encontrado_flag:
        pedido_lista.append({"item": item_nome, "quantidade": quantidade})
    # Esta função não deve retornar jsonify


def handle_order(message_param, contexto_param): # Renomeados para evitar conflito
    """Processa o pedido do usuário."""
    itens_pedido = parse_order(message_param)
    if not itens_pedido:
        # Não retorna jsonify aqui, a função principal `chat` fará isso.
        # Poderia retornar um status ou uma flag para a função `chat`
        return {"status": "falha", "mensagem": "Não entendi o que você gostaria de pedir. Pode repetir?"}


    if "pedido" not in contexto_param:
        contexto_param["pedido"] = []

    for item_novo in itens_pedido:
        nome_item_novo = item_novo["item"] # Já é o nome canônico de MENU_OPCOES devido ao parse_order
        quantidade_nova = item_novo["quantidade"]
        adicionar_item_ao_pedido(contexto_param["pedido"], nome_item_novo, quantidade_nova)
    
    # Após adicionar, é importante combinar itens duplicados se adicionar_item_ao_pedido não fizer isso completamente
    contexto_param["pedido"] = combinar_itens_pedido(contexto_param["pedido"])

    # Não retorna jsonify aqui, a função principal `chat` fará isso.
    return {"status": "sucesso", "mensagem": "Pedido atualizado."}


@app.route("/chat", methods=["POST"])
def chat():
    session_id = request.remote_addr
    # CORREÇÃO: A variável 'message' será definida dentro dos blocos if/elif de estado
    # message_input = request.json.get("message", "").strip().lower()

    if session_id not in conversations:
        conversations[session_id] = {"estado": "aguardando_cpf", "cadastro": {}, "pedido": []}
        return jsonify({"response": "Olá! Para começar, por favor me informe seu CPF."})

    contexto_atual = conversations[session_id] # Renomeado
    estado = contexto_atual["estado"]
    message_input = request.json.get("message", "").strip().lower() # Definido aqui para uso geral

    if estado == "aguardando_cpf":
        if len(message_input) != 11 or not message_input.isdigit():
            return jsonify({"response": "CPF inválido. Por favor, informe um CPF com 11 dígitos numéricos."})
        contexto_atual["cadastro"]["cpf"] = message_input
        user = find_user_by_cpf(message_input)
        if user:
            contexto_atual["nome"] = user["nome"]
            contexto_atual["cadastro"] = {
                "cpf": user["cpf"],
                "nome": user["nome"],
                "celular": user["celular"],
                "endereco": user.get("endereco", "")
            }
            contexto_atual["estado"] = "conversando"
            all_orders = load_orders()
            # Carrega o último pedido do usuário, se houver.
            # A estrutura em orders.json é {cpf: [lista_de_pedidos_passados]} ou {cpf: ultimo_pedido_lista_de_itens}
            # Se for uma lista de pedidos passados, precisa pegar o mais recente ou decidir a lógica.
            # Se for o último pedido (lista de itens), ok.
            # Assumindo que orders[cpf] é UMA lista de itens (o último pedido)
            pedido_anterior_lista_itens = all_orders.get(contexto_atual["cadastro"]["cpf"], [])
            contexto_atual["pedido"] = pedido_anterior_lista_itens # Sobrescreve o pedido atual com o anterior
            
            return jsonify({"response": f"Olá, {user['nome']}! Já te conheço pelo CPF. Como posso ajudar hoje?"})
        else:
            contexto_atual["estado"] = "aguardando_nome"
            return jsonify({"response": "CPF não encontrado. Por favor, qual o seu nome?"})

    elif estado == "aguardando_nome":
        contexto_atual["cadastro"]["nome"] = message_input.title()
        contexto_atual["estado"] = "aguardando_celular"
        return jsonify({"response": "Qual seu número de celular?"})

    elif estado == "aguardando_celular":
        contexto_atual["cadastro"]["celular"] = message_input
        contexto_atual["estado"] = "aguardando_endereco"
        return jsonify({"response": "Qual é o seu endereço completo para entrega?"})

    elif estado == "aguardando_endereco":
        contexto_atual["cadastro"]["endereco"] = message_input
        users = load_users()
        # Evitar duplicar usuários
        user_existe = False
        for i, u_existing in enumerate(users):
            if u_existing.get("cpf") == contexto_atual["cadastro"]["cpf"]:
                users[i] = contexto_atual["cadastro"] # Atualiza o usuário existente
                user_existe = True
                break
        if not user_existe:
            users.append(contexto_atual["cadastro"])
        save_users(users)
        
        nome_cadastrado = contexto_atual["cadastro"]["nome"] # Renomeado
        contexto_atual["nome"] = nome_cadastrado
        contexto_atual["estado"] = "conversando"
        contexto_atual["pedido"] = [] # Limpa o pedido ao concluir novo cadastro
        return jsonify({"response": f"Cadastro concluído, {nome_cadastrado}! Como posso ajudar você hoje?"})

    elif estado == "conversando":
        if message_input in ["ver pedido", "mostrar pedido", "meu pedido"]:
            return jsonify({"response": visualizar_pedido(contexto_atual["pedido"])}) # Usa visualizar_pedido

        elif message_input.startswith("remover item"):
            item_info = message_input[len("remover item"):].strip()
            parts = item_info.split()
            if not parts:
                return jsonify({"response": "Por favor, especifique o item a ser removido."})

            item_nome_remover = parts[0]
            quantidade_remover = None
            if len(parts) > 1 and parts[1].isdigit():
                quantidade_remover = int(parts[1])

            removidos_count = remover_item_do_pedido(contexto_atual["pedido"], item_nome_remover, quantidade_remover)
            if removidos_count > 0:
                return jsonify({"response": f"{removidos_count} {item_nome_remover}(s) removido(s) do pedido."})
            else:
                return jsonify({"response": f"Não encontrei {item_nome_remover} no seu pedido ou quantidade insuficiente."})
        
        elif message_input.startswith("editar item"):
            parts = message_input.split()
            if len(parts) >= 4 and parts[-2] == "quantidade" and parts[-1].isdigit():
                item_nome_editar = " ".join(parts[2:-2])
                nova_quantidade_editar = int(parts[-1])
                
                item_canonico_editar = None
                for key_menu_edit, val_menu_edit in MENU_OPCOES.items():
                     if remover_acentos(key_menu_edit.lower()) == remover_acentos(item_nome_editar.lower()):
                          item_canonico_editar = val_menu_edit
                          break
                if not item_canonico_editar:
                     item_canonico_editar = item_nome_editar

                encontrado_editar = False
                for item_pedido_edit in contexto_atual["pedido"]:
                    if item_pedido_edit["item"] == item_canonico_editar:
                        if nova_quantidade_editar > 0:
                             item_pedido_edit["quantidade"] = nova_quantidade_editar
                        else: # Remove o item se a nova quantidade for zero ou menos
                             contexto_atual["pedido"].remove(item_pedido_edit)
                        encontrado_editar = True
                        break
                if encontrado_editar:
                   return jsonify({"response": f"Quantidade de {item_nome_editar} atualizada para {nova_quantidade_editar}." if nova_quantidade_editar > 0 else f"{item_nome_editar} removido do pedido."})
                else:
                   return jsonify({"response": f"Não encontrei {item_nome_editar} no seu pedido."})
            else:
              return jsonify({"response": "Para editar, use: 'editar item [nome do item] quantidade [nova quantidade]'."})
        
        elif message_input.startswith("editar endereço"):
            novo_end = message_input[len("editar endereço"):].strip()
            if novo_end:
                contexto_atual["cadastro"]["endereco"] = novo_end
                # Atualizar no users.json também
                users_lista = load_users()
                for i, u_e in enumerate(users_lista):
                    if u_e.get("cpf") == contexto_atual["cadastro"]["cpf"]:
                        users_lista[i]["endereco"] = novo_end
                        save_users(users_lista)
                        break
                return jsonify({"response": f"Endereço atualizado para: {novo_end}"})
            else:
                return jsonify({"response": "Para editar endereço, use: 'editar endereço [novo endereço]'."})

        # Lógica principal de adição de itens
        # Chamada a handle_order foi removida, parse_order e adicionar_item_ao_pedido são usados diretamente
        novos_itens_detectados = parse_order(message_input)
        if novos_itens_detectados:
            for item_ni in novos_itens_detectados:
                 # Adicionar_item_ao_pedido espera o nome canônico de MENU_OPCOES, que parse_order já retorna
                adicionar_item_ao_pedido(contexto_atual["pedido"], item_ni["item"], item_ni["quantidade"])
            
            contexto_atual["pedido"] = combinar_itens_pedido(contexto_atual["pedido"]) # Garante que está combinado
            
            # Salvar o pedido atualizado em orders.json
            # append_order foi modificado para salvar o pedido completo atual
            if "cpf" in contexto_atual["cadastro"]:
                 append_order(contexto_atual["cadastro"]["cpf"], contexto_atual["pedido"])

            contexto_atual["estado"] = "aguardando_confirmacao"
            return jsonify({"response": visualizar_pedido(contexto_atual["pedido"]) + "\nVocê confirma esse pedido? (sim/não)"})

        # Tratamento de "confirmar pedido" e similares
        elif message_input in ["confirmar pedido", "finalizar pedido", "quero finalizar",
                           "confirmar minha compra", "concluir pedido", "terminar pedido",
                           "fechar pedido"]:
            if not contexto_atual["pedido"]:
                return jsonify({"response": "Seu pedido está vazio. Adicione itens antes de finalizar."})
            contexto_atual["estado"] = "aguardando_confirmacao_pagamento"
            return jsonify({
                "response": visualizar_pedido(contexto_atual["pedido"]) +
                        "\nConfirma este pedido para prosseguir com o pagamento? (sim/não)"
            })

        # Fallback para intents de IA se nenhuma ação de pedido foi detectada
        else:
            resposta_ia, tag_ia = predict_intent(message_input)
            # Lógica específica para tag de cardápio
            if tag_ia == "ver_cardapio":
                 return handle_menu_request() # Chama a função que formata o menu
            return jsonify({"response": resposta_ia})


    elif estado == "aguardando_confirmacao":
        if message_input in ["sim", "quero", "sim, quero", "confirmar", "quero confirmar"]:
            # O pedido já está como o usuário quer, apenas muda o estado
            contexto_atual["estado"] = "aguardando_forma_pagamento"
            return jsonify({"response": "Ok, pedido confirmado. Qual a forma de pagamento: dinheiro, cartão ou pix?"})
        elif message_input in ["não", "nao", "não quero", "nao quero", "cancelar", "quero cancelar"]:
            contexto_atual["estado"] = "aguardando_edicao"
            return jsonify({"response": "Ok, o que você gostaria de fazer? (cancelar, adicionar, editar, remover)"})
        else:
            return jsonify({"response": "Por favor, responda 'sim' para confirmar ou 'não' para editar o pedido."})

    elif estado == "aguardando_edicao":
        if message_input in ["cancelar", "quero cancelar", "cancelar pedido"]:
            contexto_atual["pedido"] = []
            if "cpf" in contexto_atual["cadastro"]: # Salva o pedido vazio
                append_order(contexto_atual["cadastro"]["cpf"], contexto_atual["pedido"])
            contexto_atual["estado"] = "conversando"
            return jsonify({"response": "Pedido cancelado. Se precisar de algo mais, é só dizer."})
        elif message_input in ["adicionar", "quero adicionar", "adicionar itens"]:
            contexto_atual["estado"] = "conversando"
            return jsonify({"response": "Ok, o que você gostaria de adicionar?"})
        # Para editar ou remover, o usuário precisará especificar o item, voltando para o estado "conversando"
        elif message_input.startswith("editar") or message_input.startswith("remover"):
            contexto_atual["estado"] = "conversando" # Volta para conversando para processar edição/remoção
            # Simula o reenvio da mensagem para o estado "conversando" ou instrui o usuário
            # Esta parte pode precisar de refinamento para uma melhor UX
            return jsonify({"response": f"Ok, para '{message_input}', por favor, diga novamente o comando completo (ex: editar pizza quantidade 2 ou remover pizza)."})
        elif message_input in ["editar", "quero editar", "editar itens", "editar quantidade", "editar quantidades"]:
            contexto_atual["estado"] = "conversando" # Volta para conversando para editar
            return jsonify({"response": "Ok, qual item e qual quantidade você gostaria de editar? (ex: editar pizza quantidade 2)"})

        elif message_input in ["remover", "quero remover", "remover itens", "remover item"]:
            contexto_atual["estado"] = "conversando" # Volta para conversando para remover
            return jsonify({"response": "Ok, qual item você gostaria de remover?"})
        else:
            return jsonify({"response": "Por favor, escolha: cancelar, adicionar, ou diga o que quer editar ou remover."})


    elif estado == "aguardando_edicao_quantidade": # Este estado pode não ser mais necessário com a refatoração
        contexto_atual["estado"] = "conversando"
        return jsonify({"response": "Ok, qual item e qual quantidade você gostaria de editar? (ex: editar pizza quantidade 2)"})

    elif estado == "aguardando_remocao": # Este estado pode não ser mais necessário
        contexto_atual["estado"] = "conversando"
        return jsonify({"response": "Ok, qual item você gostaria de remover?"})

    elif estado == "aguardando_confirmacao_pagamento":
        if message_input in ["sim", "quero", "sim, quero", "quero confirmar", "quero pagar"]:
            contexto_atual["estado"] = "aguardando_forma_pagamento"
            return jsonify({"response": "Ok, pedido confirmado. Qual a forma de pagamento: dinheiro, cartão ou pix?"})
        elif message_input in ["não", "nao", "não quero", "nao quero", "cancelar", "quero cancelar"]:
            # Não cancela o pedido aqui, apenas volta para edição
            contexto_atual["estado"] = "aguardando_edicao" # Ou "conversando" para adicionar/remover
            return jsonify({"response": "Ok, o que você gostaria de fazer com o pedido? (cancelar, adicionar, editar, remover)"})
        else:
            return jsonify({"response": "Por favor, responda 'sim' para confirmar ou 'não' para alterar o pedido."})

    elif estado == "aguardando_forma_pagamento":
        if message_input in ["cartão", "cartao", "pix", "dinheiro"]:
            contexto_atual["forma_pagamento"] = message_input
            contexto_atual["estado"] = "pedido_finalizado"
            
            now = datetime.datetime.now()
            timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S") # Renomeado
            
            user_info_comp = contexto_atual["cadastro"] # Renomeado
            pedido_info_comp = contexto_atual["pedido"] # Renomeado
            forma_pagamento_comp = contexto_atual["forma_pagamento"] # Renomeado

            # Montar string dos itens para o comprovante
            itens_str_comp = ""
            total_pedido_comp = 0
            for item_comp in pedido_info_comp:
                nome_item_comp = item_comp["item"]
                qtd_comp = item_comp["quantidade"]
                
                chave_preco_comp = remover_acentos(nome_item_comp.lower())
                # Similar a visualizar_pedido, garantir que a chave de preço seja correta
                preco_unit_comp = menu_prices.get(chave_preco_comp, 0) 
                
                # Fallback se o nome direto não funcionar, tentar mapear pelo menu_data
                if preco_unit_comp == 0:
                    for item_md_comp in menu_data:
                        if remover_acentos(item_md_comp["item"].lower()) == chave_preco_comp:
                            preco_unit_comp = item_md_comp["price"]
                            break
                
                subtotal_comp = preco_unit_comp * qtd_comp
                total_pedido_comp += subtotal_comp
                itens_str_comp += f"{qtd_comp}x {nome_item_comp} - R$ {preco_unit_comp:.2f} = R$ {subtotal_comp:.2f}\n"

            comprovante = f"""
            ===== Comprovante de Pagamento =====
            Data/Hora: {timestamp_str}

            --- Dados do Cliente ---
            Nome: {user_info_comp.get("nome", "N/A")}
            CPF: {user_info_comp.get("cpf", "N/A")}
            Celular: {user_info_comp.get("celular", "N/A")}
            Endereço: {user_info_comp.get("endereco", "N/A")}

            --- Pedido ---
            {itens_str_comp if itens_str_comp else "Nenhum item no pedido."}
            --- Total ---
            Total: R$ {total_pedido_comp:.2f}

            Forma de Pagamento: {forma_pagamento_comp.capitalize()}

            Obrigado!
            """
            
            # Salvar o pedido finalizado (opcional, pois já foi salvo ao adicionar itens)
            # Se quiser salvar um status de "finalizado", a estrutura de orders.json precisaria mudar
            # append_order(contexto_atual["cadastro"]["cpf"], contexto_atual["pedido"]) # Pedido já está salvo

            status_pedido_msg = random.choice(["Em preparo", "A caminho", "Saiu para entrega"])
            comprovante += f"\nStatus do Pedido: {status_pedido_msg}"

            # Limpar pedido e estado para uma nova interação (opcional, ou voltar para saudação)
            # conversations[session_id] = {"estado": "aguardando_cpf", "cadastro": {}, "pedido": []} # Reseta tudo
            contexto_atual["pedido"] = [] # Limpa o pedido atual
            contexto_atual["estado"] = "conversando" # Ou um estado de "agradecimento_final"

            return jsonify({"response": f"Pagamento via {message_input} confirmado. \n{comprovante}"})
        else:
            return jsonify({"response": "Forma de pagamento inválida. Por favor, escolha: cartão, pix ou dinheiro."})

    # CORREÇÃO: Os blocos abaixo (finalizando, pagamento, aguardando_pagamento) parecem ser lógicas mais antigas
    # ou alternativas que podem conflitar com os estados já definidos.
    # É importante ter um fluxo de estados coeso.
    # Comentando-os por enquanto, pois os estados acima cobrem o fluxo.

    # elif estado == "finalizando":
    #     # ... (lógica original) ...
    # elif estado == "pagamento":
    #     # ... (lógica original) ...
    # elif estado == "aguardando_pagamento": # Este estado é muito similar a "aguardando_forma_pagamento"
    #     # ... (lógica original) ...


    # CORREÇÃO: Lógica de fallback e tratamento de mensagem vazia
    # A variável user_message não está definida neste escopo mais externo. Usar message_input.
    if not message_input and estado != "aguardando_cpf": # Não reiniciar se estiver aguardando CPF e a mensagem for vazia
        return jsonify({"response": "Mensagem vazia.", "action": "restart"})

    # O 'intent = classify_text(user_message)' e o fallback genérico devem ser o último recurso
    # se nenhum estado específico tratar a mensagem.
    # No entanto, a lógica de estados já deve cobrir a maioria dos casos.
    # Se chegou aqui, é porque um estado não tratou a mensagem ou não retornou.

    # Fallback final se nenhum estado tratou
    # Mas o estado "conversando" já tem um fallback para predict_intent
    # Este return pode ser inalcançável ou indicar um problema no fluxo de estados
    # if 'response_text' not in locals() or not response_text: # Se nenhuma resposta foi gerada
    #     return jsonify({"response": "Desculpe, algo inesperado aconteceu. Por favor, comece uma nova conversa.", "action": "restart"})
    
    # Se o código chegar aqui, é um erro de lógica nos estados.
    # Cada if/elif de estado deve retornar um jsonify.
    # A função predict_intent no estado "conversando" é o fallback principal.
    print(f"AVISO: Chegou ao fim da função chat sem retornar. Estado: {estado}, Mensagem: {message_input}")
    return jsonify({"response": "Desculpe, ocorreu um erro interno. Tente novamente.", "action": "restart"})


# A função fechar_pedido não está sendo chamada no fluxo principal.
# A lógica de comprovante foi integrada no estado "aguardando_forma_pagamento".
# Se for necessário usá-la, ela precisaria ser chamada de algum estado.
# def fechar_pedido(contexto_param_fechar):
#     """Gera o resumo do pedido e calcula o valor total."""
#     # ... (código da função como estava, mas precisa de revisão de variáveis e consistência com menu_prices)


if __name__ == "__main__":
    app.run(debug=True)