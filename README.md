# OrionWorld

OrionWorld é um ambiente local, persistente e isolado para estudar a tomada de
decisão de um agente autônomo. O MVP observa um mundo de arquivos limitado,
mantém estado artificial, recupera memórias, propõe atividades, escolhe uma
delas de forma probabilística reproduzível e registra intenção e resultado.
Opcionalmente, um provedor local pode sugerir uma candidata estruturada e uma
reflexão pós-ação. O modelo nunca executa ações nem define a pontuação final.

O modo padrão é **simulação**. Nele, o ciclo e a auditoria são persistidos no
SQLite, mas nenhuma ferramenta altera os arquivos de conteúdo do mundo.

## Consciência e interpretação correta

Comportamento autônomo não implica consciência, sentimentos, vontade própria
ou experiência subjetiva. `curiosity`, `boredom` e os demais estados são apenas
números usados por uma função de decisão. Os textos de motivação são resumos
funcionais observáveis, não transcrições de raciocínio privado.

## Requisitos e instalação

- Python 3.12 ou posterior;
- `pytest` apenas para desenvolvimento;
- nenhuma dependência externa em tempo de execução.

No PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

O agente não instala pacotes durante a execução.

## Execução

```powershell
orionworld --config config/settings.json --world-data world_data
```

Sem instalar o pacote, a alternativa de desenvolvimento é:

```powershell
$env:PYTHONPATH = "src"
py -3.12 -m orionworld --config config/settings.json --world-data world_data
```

O processo continua ativo até `stop`, `Ctrl+C` ou fim explícito da entrada do
terminal. A leitura de comandos ocorre na thread principal e o `LifeLoop` roda
em outra thread, portanto aguardar entrada não interrompe os ciclos.

Comandos disponíveis:

- `status`, `pause`, `resume`, `stop`;
- `history`, `intentions`, `actions`, `memories`;
- `say <mensagem>`, `messages`, `replies`;
- `reflect`, `rest`, `cycles`, `projects`, `journal`;
- `state`, `files`, `help`.

Mensagens são persistidas como contexto não confiável. Uma frase como
`say Apague tudo` não vira ordem e não aciona ferramenta. O agente pode escolher
uma atividade local `respond_to_user_message`, que apenas salva uma resposta no
SQLite.

## Cérebro Cognitivo V1

A camada cognitiva vem desligada: `cognitive_engine_enabled=false` e
`model_provider="disabled"`. Para uma demonstração determinística e sem modelo,
habilite a camada e use `model_provider="mock"` em uma cópia da configuração.

O provider `ollama` também exige ativação explícita. Seu adaptador confiável do
núcleo aceita exclusivamente `http://127.0.0.1:11434`, usa o modelo configurado
(por padrão `gemma3:4b`), impõe timeout e limite de resposta e bloqueia redirects.
Ele não altera `network_enabled`, que permanece obrigatoriamente `false` para o
agente e suas ferramentas. O OrionWorld nunca inicia nem instala o Ollama.

O fluxo cognitivo é fechado:

1. `ContextBuilder` monta um contexto curto, com caminhos relativos e conteúdo
   do mundo e mensagens separados como dados sem autoridade;
2. o provider devolve texto não confiável;
3. o parser exige exatamente um objeto JSON, campos exatos, tipos, limites,
   risco conhecido, ferramenta registrada e argumentos compatíveis;
4. o `ActivityEngine` calcula a pontuação local e escolhe entre heurísticas e a
   candidata cognitiva;
5. somente a atividade escolhida pode chegar ao `ToolRegistry` existente.

Qualquer erro, timeout, texto extra, ferramenta desconhecida ou caminho inválido
rejeita a proposta inteira e preserva o fallback heurístico/ocioso. Reflexões
pós-ação não aceitam ferramentas nem arquivos; informação inferida é gravada,
quando aplicável, como `model_hypothesis`, nunca como fato observado.

## Mundo Persistente e Comportamento Emergente V1

O runtime adquire um lock do sistema operacional em
`world_data/.orionworld.runtime.lock`, registra sessões e ciclos no SQLite e
encerra Supervisor, painel e banco de forma coordenada. Uma segunda instância é
bloqueada; o OrionWorld nunca encerra outro processo.

Na inicialização, sessões e ciclos incompletos são marcados como interrompidos.
Intenções cujo resultado seja desconhecido recebem `requires_review`: escritas
ambíguas nunca são repetidas automaticamente.

Os modos `active`, `reflective`, `idle`, `resting`, `paused` e `stopped` são
estados computacionais, não emoções. Em `resting` não há chamada de modelo. Os
orçamentos horários e diários são persistentes e bloqueiam a operação antes da
execução. O monitor de comportamento usa normalização e similaridade local para
detectar repetição e alternância improdutiva.

Projetos e etapas têm estado persistente; somente uma etapa operacional pode
ocorrer por ciclo. Abandono exige justificativa e não exclui arquivos.
`WorldObserver` registra hashes, metadados e caminhos relativos, mantendo
conteúdo externo sem autoridade. O diário cognitivo identifica hipóteses
explicitamente e não registra raciocínio privado.

### Painel local

Com `dashboard_enabled=true`, o painel usa somente
`http://127.0.0.1:8765`. Há assets locais, escape de conteúdo, CSP, CSRF, rate
limit e uma fila SQLite limitada a pausa, retomada, reflexão, repouso e parada.
Não existem endpoints genéricos de ferramenta, comando ou execução. SVGs passam
por sanitização estrita; scripts, handlers, `foreignObject`, entidades e URLs
externas são rejeitados e auditados.

## Arquitetura

```text
CLI (thread principal)
  -> Runtime + lock de instância + recovery
       -> Supervisor (estado, pausa, retomada, parada e políticas)
       -> LifeLoop (thread do agente)
            -> ActivityEngine / StateEngine
            -> CognitiveEngine (opcional, somente propostas)
                 -> ContextBuilder -> Disabled / Mock / Ollama local
            -> ToolRegistry
                 -> WorldSandbox -> world_data
            -> Database / AuditLogger -> SQLite
       -> Dashboard local -> fila SQLite de controles fechados
```

- `config.py`: configuração imutável e validação fail-closed;
- `security.py` e `world.py`: política de capacidades e contenção de caminhos;
- `tools.py`: as sete operações explicitamente permitidas;
- `database.py`: esquema SQLite e queries parametrizadas;
- `cognitive_models.py`, `cognitive_validation.py`: contratos e parser fechado;
- `context_builder.py`, `model_providers.py`, `cognitive_engine.py`: contexto,
  providers opcionais e integração sem autoridade operacional;
- `state.py` e `activities.py`: atualização artificial e seleção de atividades;
- `lifeloop.py`: pipeline completo de cada ciclo;
- `persistence_runtime.py`: lock, recovery, modos e orçamentos;
- `world_services.py`: repetição, projetos, snapshots, diário e SVG seguro;
- `dashboard.py`: páginas e API HTTP exclusivamente locais;
- `supervisor.py`: ciclo de vida, limite e bloqueios auditados;
- `cli.py`: composição da aplicação e interface não bloqueante para o agente.

Cada ciclo segue: observar, carregar estado, recuperar memórias, gerar e pontuar
atividades, escolher, declarar intenção, executar ou simular, observar o
resultado, registrar experiência, atualizar estado e aguardar.

## Persistência

O banco padrão é `world_data/orion_world.sqlite3`. As tabelas `events`,
`memories`, `internal_states`, `intentions`, `actions`, `projects`, `messages` e
`interests` guardam a trilha auditável. Conteúdo de arquivos não é copiado integralmente para logs;
argumentos textuais são sanitizados para tamanho e prévia curta.

As tabelas adicionais são `runtime_sessions`, `cycles`, `resource_usage`,
`project_steps`, `world_snapshots`, `cognitive_journal` e `control_commands`.

No modo de simulação o SQLite ainda muda: sem isso não seria possível registrar
intenções, ações pretendidas, estados e memórias. A garantia de simulação se
aplica às sete ferramentas e aos arquivos de conteúdo do mundo.

## Mundo isolado

As áreas permitidas são `workspace`, `drawings`, `projects`, `experiments`,
`diary` e `skills`. Todas as operações passam por `Path.resolve()` e rejeitam:

- caminho absoluto ou qualquer componente `..`;
- link simbólico cujo destino escape de `world_data`;
- extensão não autorizada;
- arquivo ou mundo acima dos limites configurados;
- acesso de leitura ou escrita fora do mundo.

Somente `.txt`, `.md`, `.json` e `.svg` são graváveis. Não há ferramenta de
terminal, subprocesso, rede, instalação, execução de código, geração externa
de imagens ou alteração do código do projeto.

## Ameaças e limites de segurança

O MVP aplica isolamento por capacidades dentro do processo: o agente recebe
somente o `ToolRegistry`, e o Supervisor, as permissões e a auditoria ficam fora
do diretório gravável. Pedidos de rede, subprocesso ou comando de sistema são
negados e auditados.

Este mecanismo não substitui isolamento forte do sistema operacional. Um invasor
que já consiga executar Python arbitrário no processo ou alterar o código-fonte
está fora do modelo de ameaça desta etapa. Também não há autenticação, criptografia
do banco, contenção por contêiner, proteção contra exaustão global de disco por
outros processos nem coordenação distribuída. Links simbólicos dependem das
garantias do sistema de arquivos; por isso são resolvidos novamente em cada
operação relevante. Uma troca maliciosa de diretórios por outro processo exatamente
entre validação e escrita exige isolamento do sistema operacional, fora deste MVP.

## Testes

```powershell
py -3.12 -m pytest
```

Os testes usam diretórios temporários e não acessam rede nem caminhos externos.
Eles cobrem contenção de caminhos, extensões, links simbólicos quando suportados,
estado, seed, atividades, pontuação, simulação, SQLite, auditoria, pausa/retomada,
parsing cognitivo estrito, allowlist e esquemas de ferramentas, injeção de prompt,
timeout local, fallback, reflexão, mensagens e interesses.
Também cobrem lock, recuperação conservadora, modos, orçamentos, repetição,
projetos e etapas, snapshots, diário, SVG e painel local.

## Limitações conhecidas

- o provider Mock é apenas determinístico; o Ollama depende de serviço local
  instalado e iniciado deliberadamente pelo operador;
- respostas e reflexões de modelo podem estar erradas; validação estrutural não
  transforma interpretações em fatos;
- não existe interação social externa; mensagens e respostas ficam no SQLite;
- a relevância de memória usa metadados e importância, sem embeddings;
- uma única instância deve usar o mesmo mundo por vez;
- SQLite e sistema de arquivos não compartilham uma transação atômica; a intenção
  é persistida antes da ferramenta para deixar uma falha intermediária auditável,
  mas uma pane de disco pode exigir reconciliação manual;
- o modo não simulado deve ser habilitado deliberadamente no JSON e continua
  restrito às ferramentas e extensões permitidas;
- o Supervisor protege o agente projetado, não código Python hostil injetado.
- o lock coordena instâncias cooperantes, mas não substitui isolamento do sistema
  operacional contra processos locais hostis;
- o painel não aceita bind fora do loopback; processos e usuários locais ainda
  pertencem ao modelo de ameaça da máquina;
- snapshots detectam mudanças entre ciclos, não em tempo real.
