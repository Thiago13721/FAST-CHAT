# produto_utils.py

def identificar_produto_por_texto(input_usuario, menu):
    input_lower = input_usuario.lower()
    
    # Criar uma lista de todos os aliases com o produto correspondente
    # e ordená-los por tamanho (do mais longo para o mais curto)
    # Isso garante que "pizza de calabresa" seja verificado antes de "calabresa"
    all_aliases = []
    for produto in menu:
        for alias in produto.get("aliases", []):
            all_aliases.append((alias.lower(), produto)) # Armazena o alias em minúsculas e o produto
    
    # Ordena os aliases do mais longo para o mais curto
    all_aliases.sort(key=lambda x: len(x[0]), reverse=True)

    for alias, produto in all_aliases:
        # Verifica se o alias (agora em minúsculas) está na entrada do usuário
        if alias in input_lower:
            return produto
            
    return None

def classify_text(texto):
    """
    Função fictícia para classificar texto. 
    Use uma IA ou modelo treinado futuramente.
    """
    texto = texto.lower()

    if "pedido" in texto or "quero" in texto:
        return "pedido"
    elif "cardápio" in texto or "menu" in texto:
        return "ver_cardapio"
    elif "obrigado" in texto:
        return "agradecimento"
    elif "tchau" in texto:
        return "despedida"
    else:
        return "indefinido"
