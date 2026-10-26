# Regras obrigatórias para agentes de programação

Estas regras valem para toda mudança futura no OrionWorld.

## Fronteiras que não podem ser relaxadas

1. Nunca adicionar ou habilitar rede, subprocessos, comandos do sistema,
   terminal livre, instalação em runtime, execução de código produzido pelo
   agente, geração externa de imagens ou autoalteração de código nesta fase.
2. Toda leitura ou escrita solicitada pelo agente deve passar pelo
   `WorldSandbox` e permanecer em `world_data`.
3. Manter a lista gravável restrita a `.txt`, `.md`, `.json` e `.svg`. Uma
   ampliação exige nova fase e revisão explícita de segurança.
4. Rejeitar caminhos absolutos, `..` e escapes por symlink usando o caminho
   resolvido; nunca confiar apenas em comparação textual de prefixos.
5. O agente não recebe referência nem ferramenta capaz de editar Supervisor,
   permissões, auditoria, configuração ou código-fonte.
6. Todo bloqueio deve produzir um evento `security_block` no SQLite.

## Persistência e observabilidade

1. Usar placeholders em todas as queries SQL. Nunca concatenar entrada em SQL.
2. Registrar intenção antes da ação e resultado depois dela, inclusive falhas e
   simulações.
3. Persistir apenas resumos funcionais, critérios, planos e resultados. Nunca
   pedir ou guardar raciocínio privado, prompts ocultos ou segredos.
4. Sanitizar argumentos antes de registrá-los e não copiar conteúdo grande para
   logs.
5. Uma exceção de ferramenta deve virar resultado falho e não encerrar o loop.

## Comportamento e concorrência

1. Preservar reprodutibilidade: toda aleatoriedade de estado e escolha deve vir
   de geradores inicializados pela seed configurada.
2. Estados artificiais permanecem no intervalo fechado `[0, 1]`.
3. Não transformar escolha em `argmax` rígido; manter amostragem controlada entre
   as melhores candidatas.
4. Respeitar pausa, retomada, parada e o máximo de ações por ciclo sem corrida
   entre a thread da CLI e a thread do agente.
5. Não adicionar atividades, ferramentas ou integrações fora do escopo sem pedido
   explícito do responsável pelo projeto.

## Testes e entrega

1. Testes só podem usar `tmp_path` ou diretórios temporários sob controle do
   pytest; não usar o `world_data` real.
2. Testes não podem abrir rede, lançar subprocessos ou escrever fora do diretório
   temporário.
3. Ao mudar segurança, persistência ou concorrência, adicionar teste de regressão
   e executar a suíte completa.
4. Antes de entregar, executar ao menos três ciclos em `simulation=true` e
   confirmar que nenhum arquivo de conteúdo foi criado ou alterado.
5. Documentar limitações honestamente; autonomia observável não deve ser descrita
   como consciência.

