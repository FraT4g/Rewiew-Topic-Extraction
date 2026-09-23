## Descrizione del progetto e fonte dei dati
Il progetto si concentra sull'implementazione dell'algoritmo di Topic Extraction basato sul modello probabilistico Latent Dirichlet Allocation (LDA), applicato a un dataset di recensioni in lingua italiana pubblicate dagli utenti sulla piattaforma Tripadvisor, con l'obiettivo di individuare gli argomenti principali trattati nelle recensioni.

[link al dataset scaricabile da Kaggle](https://www.kaggle.com/datasets/alessandrolobello/italian-tripadvisor)

## Background teorico
### Approccio bayesiano
**Formula di Bayes**

$$ \mathbb{P}(\theta \mid X )=\frac{\mathbb{P}(X \mid\theta)P(\theta)}{\mathbb{P}(X)} $$

 dove:
 * $\mathbb{P}(\theta \mid X )$ è la Posterior distribution e rappresenta la probabilità a posteriori (dopo aver preso in considerazione i dati osservati) di $\theta$
 * $\mathbb{P}(\theta)$ è la prior distribution e definisce le ipotesi iniziali sul parametro $\theta$
 * $\mathbb{P}(X \mid \theta)$ è la cosidetta verosimiglianza e rappresenta la probabilità di
 osservare i dati assumendo il parametro $\theta$
 * $\mathbb{P}(X)$ è la probabilità di X che si può esprimere, condizionando su tutti i possibili valori che può assumere $\theta$, con la seguente formula:
 $$\mathbb{P}(X)=\int \mathbb{P}(X \mid \theta)\mathbb{P}(\theta) d\theta $$

 Quest' ultimo integrale, è spesso difficile da computare a causa dell'elevata dimensionalità dello spazio dei parametri. Per ovviare a questo problema, si utilizzano tecniche di stima che permettono, con buona approssimazione, di ottenere ugualmente un' espressione per la posterior distribution.
 ### Cenni su catene di Markov e metodo Montecarlo
 
**Catene di Markov**

Una **catena di Markov** è un processo stocastico costituito da una sequenza di variabili aleatorie $(X_n)_n$, che assumono valori in un certo spazio degli stati e che evolve nel tempo in accordo alla **proprietà di Markov**, ovvero:

Per ogni istante di tempo $n$ e per ogni possibile scelta di una sequenza di stati $x_0,x_1,\dots,x_n$ tale che

$$
\mathbb{P}(X_0=x_0,\dots,X_n=x_n)>0,
$$

vale che

$$
\begin{aligned}
&\mathbb{P}(X_{n+1}=x_{n+1}\mid X_n=x_n,\dots,X_0=x_0)\\
&\qquad=\mathbb{P}(X_{n+1}=x_{n+1}\mid X_n=x_n)\\
&\qquad=P(x_n,x_{n+1}).
\end{aligned}
$$

In altre parole, il processo, all'istante $n$, sceglie la sua posizione successiva tenendo conto solamente della sua posizione attuale $X_n$, dimenticando tutte le mosse effettuate in passato.

Le catene di maggiore interesse sono quelle che ammettono una **distribuzione di probabilità stazionaria**, ossia una distribuzione $pi$ che soddisfa

$$
\pi P=\pi,
$$

vale a dire

$$
\forall x,\qquad \pi(x)=\sum_y \pi(y)P(y,x).
$$

Possiamo interpretare la definizione nella maniera seguente: se la distribuzione attuale è $pi$, effettuando una transizione della catena, rimarremo distribuiti nello stesso modo.

Queste distribuzioni assumono un ruolo centrale nello studio delle catene di Markov, poiché, sotto opportune condizioni, la distribuzione stazionaria coincide con la distribuzione asintotica della catena.

**Metodo Montecarlo e relazione con la statistica Bayesiana**

Gli algoritmi MCMC, sfruttano le proprietà delle catene di Markov con l'obiettivo di simulare il campionamento da una certa distribuzione di probabilità target $\pi$ o analogamente computare $\mathbb{E}_{\pi}(f)$, il valore atteso di una funzione sotto la distribuzione $\pi$. Infatti, l'idea su cui si basano tali algoritmi, consiste nel costruire una catena di Markov che converga a $\pi$ (quindi che abbia $\pi$ come distr. stazionaria). La ragione principale alla base dell'implementazione di questo tipo  di tecniche, risiede nel fatto che le distribuzioni target assumono spesso espressioni  complicate, che dipendono da costanti di normalizzazione molto difficili da computare.
 Gli algoritmi MCMC trovano una naturale applicazione nel contesto dell' inferenza bayesiana. In questo tipo di paradigma infatti, come abbiamo visto, l'interesse è volto ad ottenere la densità **a posteriori** del vettore di parametri $\theta$.

 ## Topic Extraction

 La classe degli algoritmi di topic extraction, è quella classe di algoritmi di Natural Language Processing (NLP) che si occupa di ricavare informazioni da un insieme molto vasto di documenti, fornendo misure quantitative che possono essere utilizzate per identificare il contenuto dei testi, ossia definire quali sono gli argomenti trattati (topic).
 Il topic extraction, è una tecnica non supervisionata.Questo significa che non effettua l'apprendimento a partire da etichette preassegnate a ciascun documento della collezione, bensì ricava le distribuzioni di probabilità sulle parole del vocabolario associate a ciascuno dei topic. Attraverso l'analisi delle parole più probabili, sta a noi risalire a quale sia l'argomento trattato.

 ## Latent Dirichlet Allocation

 Il Latent Dirichlet Allocation (LDA) è un modello **generativo**, cioè un modello che descrive e **semplifica** il **processo di creazione di un documento testuale** all'interno di un corpus , assumendo che esso avvenga secondo una serie di passaggi probabilistici.
Nello specifico, il LDA adotta un **approccio bayesiano** : costruisce un processo stocastico che si presume abbia generato i  vari testi osservati nel corpus a partire da specifiche **ipotesi a priori**.
Il termine "**latente**" si riferisce a quelle variabili nascoste che intervengono nel processo di generazione del testo e che non sono direttamente osservabili. Nel contesto del LDA, tali variabili latenti corrispondono ai **topic**, intesi come distribuzioni di probabilità sull'insieme delle parole utilizzate nei vari documenti.

 ### Descrizione del processo generativo

 Descriviamo di seguito come avviene il processo generativo nel modello LDA, scegliendo un numero fissato T di topic, un numero D di documenti nel corpus, e un numero N di parole del vocabolario
1. Per ogni topic $t$ $\in \{1 \dots \text{T}\}$:
+ (a) A partire da una distr. di Dirichlet simmetrica, si estrae il parametro di una distr. multinomiale del topic $t$ sulle le parole del vocabolario, $\phi_z \sim  \text{Dir}(\beta)$
2. Per ogni documento $d$ $\in \{1 \dots D \}$
+ (a) A partitre da una distr. di Dirichlet simmetrica si estrae il parametro di una distr. multinomiale del documento $d$ sui topic, $\theta_d \sim \text{Dir}(\alpha)$
+ (b) Per ogni parola $w$ $\in \{ 1 \dots N_d\}$
   * (i) Viene estratto un topic dalla distr. del documento $d$ sui topic, $z_{dn} \sim \text{Multinomial} (\theta_d)$
   * (ii) Viene estratta una parola dalla distr. dei topic sulle parole, $w_{dn} \sim \text{Multinomial}(\phi_z)$

  ### Osservazione
  Nel caso del modello LDA, la prior distribution è dunque la distribuzione di Dirichlet che viene usata per estarre i parametri delle distribuzioni multinomiali:      
     * $\theta_d=(\theta_{d,1} \dots \theta_{d,T} )$ è la distr. dei documenti sui topic,e ogni entrata $i$ rappresenta la prob. del topic $i$ per il documento $d$ per cui $\sum_{i=1}^T \theta_{d,i}=1$
     * $\phi_z=(\phi_{z,1} \dots \phi_{z,N} )$ è la dist. dei topic sulle parole, e ogni entrata $j$ rappresenta la prob. della $j$-esia parola per il topic $t$ per cui $\sum_{j=1}^N \phi_{z,j}=1$

  ### Formula per la probabilità congiunta

  Riportiamo di seguito l' espressione per la probabilità congiunta del modello LDA, valida sotto le ipotesi di **scambiabilità dei documenti** e **scambiabilità delle parole**.
  
  $p(\mathbf{z}, \mathbf{w}, \theta_{1:D}, \phi_{1:T} \mid \alpha, \beta)
= \prod_{t=1}^{T} p(\phi_t \mid \beta)
\prod_{d=1}^{D} p(\theta_d \mid \alpha)
 \prod_{n=1}^{N_d}
p(z_{dn} \mid \theta_d)
p(w_{dn} \mid z_{dn}, \phi_{1:T})$
  
           
        
  
  dove :   
  * $z_{dn}$ rappresenta l'assegnazione di un certo topic all' $n$-esima parola del documento d
  * $\textbf{z}$ è l'assegnazione di topic per tutte le parole
  * $\textbf{w}$ è l'insieme di tutte le parole che compaiono nel corpus di documenti

  ## Topic Extraction assumendo LDA

La messa a punto dell'algoritmo di Topic Extraction si fonda sull'inversione del processo generativo, assunto alla base della produzione del corpus documentale, con l'obiettivo di inferire le variabili latenti a partire dalle variabili osservate, rappresentate dalle sequenze di parole che compongono i diversi testi.
Le variabili latenti da determinare sono:
* $z_{dn} →$ assegnazione dei topic a ciascuna parola nei dcoumenti. Per ogni parola di un documento, vogliamo determinare a quale topic è assegnata
* $\theta_{1:D} →$ la distribuzione sui topic associata ad ogni documento $1 \dots D$
* $\phi_{1:T} →$ la distribuzione sulle parole associata ad ogni topic $1 \dots T$.

Il problema inferenziale che deve essere risolto, consiste nel calcolo della distr. a posteriori :

$$p(\textbf{z}, \theta_{1:D}, \phi_{1:T} \mid \textbf{w}, \alpha, \beta) = \frac{p(\textbf{z}, \theta_{1:D}, \phi_{1:T} , \textbf{w}\mid \alpha, \beta)}{p(\textbf{w} \mid \alpha,\beta)} $$

con la formula che è l'applicazione della formula di Bayes su più eventi:

$$\mathbb{P}(A \mid B \cap C)=\frac{ \mathbb{P}(A \cap B \cap C)} {\mathbb{P}(B \cap C)}=\frac{\mathbb{P}(A \cap B \mid C) \mathbb{P}(C)}{\mathbb{P}(B \cap C)}=\frac{\mathbb{P}(A \cap B \mid C )}{\mathbb{P}(B \mid C)} $$

Questa distribuzione, tuttavia, non è trattabile analiticamnete, essendo il denominatore ottenibile solo mediante integrazione da effettuare su uno spazio parametrico di elevata dimensionalità.
Quindi l'inferenza esatta è intrattabile, ma si può utilizzare l'inferenza approssimata utilizzando ad esempio il metodo del **Gibbs Sampling**. L'obiettivo consiste dunque nell'approssimare la distribuzione a posteriori ed ottenere una stima dei parametri tramite ottimizzazione.

## Gibbs Sampling

Il Gibbs Sampling, fa parte della classe di algoritmi del framework Markov Chain Montecarlo (MCMC) e viene usato per il **Topic Modelling**  nel contesto del LDA.

### Utlizzo del Gibbs Sampling in LDA

L'obiettivo del Gibbs sampling applicato al modello LDA, è quello di costruire una catena di Markov che abbia la posterior distribution $p(\textbf{z}, \theta_{1:D}, \phi_{1:T} \mid \textbf{w}, \alpha, \beta)$ come distr. stazionaria, in modo tale che il comportamento della catena dopo un grande numero di passi approssimi bene la distr. stazionaria (in virtu del Teorema Ergodico).
Ogni stato della catena è rappresentato dalle variabili $z_{dn}$ che rappresentano le assegnazioni dei topic per ciascuna parola. Le transizioni tra gli stati successivi, invece, avvegono tramite il campionamento sequenziale delle assegnazioni $z_{dn}$ da una distr. condizionale $P(z_{dn}=k \mid z_{-dn}, w_{dn}, d)$
dove :
* $z_{dn}$ corrisponde al topic campionato *k* assegnato alla parola $w_{dn}$
* $z_{-dn}$ rappresenta le  assegnazioni del topic in questione a tutte le altre parole.
Le variabili campionate vengono dunque modificate sequenzialmente fino a quando non viene raggiunta la distr. stazionaria della catena. Dopo aver stimato la distr. a posteriori di  $z_{dn}$, questa  viene sfruttata per dedurre le distr. $\phi_{1:T}$ e $\theta_{1:D}$.
La distr. condizionale del Gibbs Sampling è definita come:

$$ P(z_{dn}=k \mid z_{-dn}, w_{dn}, d) \propto \frac{C_{w_i,k}^1+\beta}{\sum_w C_{w,k}^1+ N\beta} \frac{C_{d,k}^2+\alpha}{\sum_t C_{d,t}^2 +T\alpha}$$

dove:
* $C^1$ è è la matrice il cui elemento $w_i,k$ rappresenta il conteggio di quante volte la parola $w_i$ è stata assegnata al topic $k$
* $C^2$ è è la matrice il cui elemento $d,k$ rappresenta il numero di volte che l'argomento $k$ è stato assegnato ad una qualsiasi parola del documento $d$
* $\sum_t C_{d,t}^2$ è il conteggio totale di assegnazioni di topic alle parole nel documento $d$
* $\sum_w C_{w,k}^1$ è il conteggio totale di parole assegnate al topic $k$.

Quindi per semplificare possiamo dire che questa espressione tiene conto di :
* Quanta "presenza di topic $k$" c'è nel documento d
* Quanto la parola $w$ partecipa al topic $k$

## Esecuzione dell'algoritmo

Per ogni documento $\{1 \dots D \}$ del corpus ripeti :

1. Si assegna a ciascuna parola $w_{dn}$ del documento un topic casuale $\{1 \dots T \}$
2. Per ciascuna parola $w_{dn}$ si iterano i seguenti passaggi:
    * Viene tolta l' assegnazione corrente della parola $w_{dn}$ all' argomento $k$.  Conseguentemente gli elementi $C_{w_i,k}^1$ e $C_{d,k}^2$ vengono decrementati di uno.
    * Un nuovo argomento viene campionato dalla distr. condizionale del Gibbs Sampling  ed assegnato alla parola $w_{dn}$: le matrici vengono aggiornate di conseguenza

Dunque, ad ogni iterazione,  si passa attraverso ciascun termine del corpus e si aggiorna l'assegnazione dei topic utilizzando una distr. condizionale basata sulle assegnazioni attuali per tutte le altre parole ($z_{-dn}$).

Al termine di un passaggio completo su tutte le parrole, si ottiene un campione di Gibbs che rappresenta lo stato corrente delle assegnazioni $z_{dn}$ per tutto il corpus di documenti. Questo processo viene ripetuto molte volte: durante la fase iniziale, detta **burn-in**, i campioni vengono scartati perchè non rappresentano accuratamente la distr. a posteriori, poi i campioni successivi, che iniziano a convergere verso la distr. target, vengono salvati a intervalli regolari per ridurre le correlazioni tra campioni consecutivi.
### Stima di $\theta_{1:D}$ e $\phi_{1:T}$
Per valutare il contenuto tematico dei documenti si richiedono le stime di $\theta_{1:D}$ e $\phi_{1:T}$. Queste grandezze possono essere facilmente ottenute a partire dai rapporti definiti in precedenza:

* $$\theta_{d,k}=\frac{C_{d,k}^2+\alpha}{\sum_t C_{d,t}^2+T\alpha} $$

* $$\phi_{k,wi}=\frac{C_{k,w_i}^1+\beta}{\sum_w C_{k,w}^1+N\beta} $$

## Prewiew del Dataset

## Esecuzione dell' algoritmo e ricostruzione dei topic

## Struttura della repository
- `notebook/` : Jupyter Notebook contenente il codice del progetto.
- `immagini/` : Selezione di immagini contenute in questo README.
## Come consultare il progetto
Il Jupyter Notebook è il punto di accesso all'intera analisi e permette di approfondire gli aspetti metodologici e implementativi del progetto. Grazie all'integrazione di codice, commenti e risultati, è possibile seguire il processo di analisi nel dettaglio, compresi i vari accorgimenti operativi e le ottimizzazioni adottate.

