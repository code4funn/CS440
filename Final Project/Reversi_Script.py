import numpy as np
import random
import matplotlib.pyplot as plt
from copy import deepcopy
from copy import copy

################################################################################################################################################# General functions applicable to all methods #######
###########################################################################################################################################

def startState(m):
    state = [' ']*(m*m)
    state[getInd(int(m/2)-1,int(m/2)-1,m)] = 'O'
    state[getInd(int(m/2)-1,int(m/2),m)] = 'X'
    state[getInd(int(m/2),int(m/2)-1,m)] = 'X'
    state[getInd(int(m/2),int(m/2),m)] = 'O'
    return state

def printState(state):
    n = len(state)
    m = n**0.5
    print('|',end='')
    for i in range(n):
        #print(state[i] if state[i] is not ' ' else i,end='|')
        print(state[i],end='|')
        if (i+1)%m==0 and i+1<n:
            print()
            print('|',end='')
    print()
    
def winner(state):
    countX = state.count('X')
    countO  = state.count('O')
    
    if countX+countO==len(state):
        if countX>countO:
            return 'X' # True # 'X'
        elif countX<countO:
            return 'O' # False # 'O'
    nValidMovesX = len(validMoves(state, 'X')) if validMoves(state,'X') != [] else 0
    nValidMovesO = len(validMoves(state, 'O')) if validMoves(state,'O') != [] else 0
    if nValidMovesX+nValidMovesO==0:
        if countX>countO:
            return 'X' # True # 'X'
        elif countX<countO:
            return 'O' # False # 'O'
        elif countX==countO:
            return 'draw'
        
def indX(state):
    l=[]
    for i in range(len(state)):
        if state[i]=='X':
            l.append(i)
    return l

def indO(state):
    l=[]
    for i in range(len(state)):
        if state[i]=='O':
            l.append(i)
    return l

def getCod(e,m):
    '''get the coordinates in 2d'''
    q = e % m
    for i in range(m):
        if e == (m*i)+q:
            p=i
    return p,q

def getInd(p,q,m):
    return (p*m)+q

def validMoves(state, player):
    n = len(state)
    m = int(n**0.5)
    moves = []
    if player=='X':
        if True: #########m==4:
            l = indX(state)
            for i in range(len(l)):
                p,q = getCod(l[i],m)
                j=1
                #check left
                while q-j>0 and state[l[i]-j]=='O':
                    j+=1
                if j!=1:
                    if state[l[i]-j]==' ' and (l[i]-j) not in moves:
                        moves.append(l[i]-j)
                    j=1
                #check right
                while q+j<m-1 and state[l[i]+j]=='O':
                    j+=1
                if j!=1:
                    if state[l[i]+j]==' ' and (l[i]+j) not in moves:
                        moves.append(l[i]+j)
                    j=1
                #check up
                while p-j>0 and state[l[i]-(j*m)]=='O':
                    j+=1
                if j!=1:
                    if state[l[i]-(j*m)]==' ' and (l[i]-(j*m)) not in moves:
                        moves.append(l[i]-(j*m))
                    j=1
                #check down
                while p+j<m-1 and state[l[i]+(j*m)]=='O':
                    j+=1
                if j!=1:
                    if state[l[i]+(j*m)]==' ' and (l[i]+(j*m)) not in moves:
                        moves.append(l[i]+(j*m))
                    j=1
                #check diagonally down left 
                while (p+j<m-1 and q-j>0) and state[l[i]+(j*(m-1))]=='O':
                    j+=1
                if j!=1:
                    if state[l[i]+(j*(m-1))]==' ' and (l[i]+(j*(m-1))) not in moves:
                        moves.append(l[i]+(j*(m-1)))
                    j=1
                #check diagonally down right 
                while (p+j<m-1 and q+j<m-1) and state[l[i]+(j*(m+1))]=='O':
                    j+=1
                if j!=1:
                    if state[l[i]+(j*(m+1))]==' ' and (l[i]+(j*(m+1))) not in moves:
                        moves.append(l[i]+(j*(m+1)))
                    j=1
                #check diagonally up left 
                while (p-j>0 and q-j>0) and state[l[i]-(j*(m+1))]=='O':
                    j+=1
                if j!=1:
                    if state[l[i]-(j*(m+1))]==' ' and (l[i]-(j*(m+1))) not in moves:
                        moves.append(l[i]-(j*(m+1)))
                    j=1
                #check diagonally up right 
                while (p-j>0 and q+j<m-1) and state[l[i]-(j*(m-1))]=='O':
                    j+=1
                if j!=1:
                    if state[l[i]-(j*(m-1))]==' ' and (l[i]-(j*(m-1))) not in moves:
                        moves.append(l[i]-(j*(m-1)))
                    j=1
            return moves
    else:
        if True:        ###########m==4:
            l = indO(state)
            for i in range(len(l)):
                p,q = getCod(l[i],m)
                j=1
                #check left
                while q-j>0 and state[l[i]-j]=='X':
                    j+=1
                if j!=1:
                    if state[l[i]-j]==' ' and (l[i]-j) not in moves:
                        moves.append(l[i]-j)
                    j=1
                #check right
                while q+j<m-1 and state[l[i]+j]=='X':
                    j+=1
                if j!=1:
                    if state[l[i]+j]==' ' and (l[i]+j) not in moves:
                        moves.append(l[i]+j)
                    j=1
                #check up
                while p-j>0 and state[l[i]-(j*m)]=='X':
                    j+=1
                if j!=1:
                    if state[l[i]-(j*m)]==' ' and (l[i]-(j*m)) not in moves:
                        moves.append(l[i]-(j*m))
                    j=1
                #check down
                while p+j<m-1 and state[l[i]+(j*m)]=='X':
                    j+=1
                if j!=1:
                    if state[l[i]+(j*m)]==' ' and (l[i]+(j*m)) not in moves:
                        moves.append(l[i]+(j*m))
                    j=1
                #check diagonally down left 
                while (p+j<m-1 and q-j>0) and state[l[i]+(j*(m-1))]=='X':
                    j+=1
                if j!=1:
                    if state[l[i]+(j*(m-1))]==' ' and (l[i]+(j*(m-1))) not in moves:
                        moves.append(l[i]+(j*(m-1)))
                    j=1
                #check diagonally down right 
                while (p+j<m-1 and q+j<m-1) and state[l[i]+(j*(m+1))]=='X':
                    j+=1
                if j!=1:
                    if state[l[i]+(j*(m+1))]==' ' and (l[i]+(j*(m+1))) not in moves:
                        moves.append(l[i]+(j*(m+1)))
                    j=1
                #check diagonally up left 
                while (p-j>0 and q-j>0) and state[l[i]-(j*(m+1))]=='X':
                    j+=1
                if j!=1:
                    if state[l[i]-(j*(m+1))]==' ' and (l[i]-(j*(m+1))) not in moves:
                        moves.append(l[i]-(j*(m+1)))
                    j=1
                #check diagonally up right 
                while (p-j>0 and q+j<m-1) and state[l[i]-(j*(m-1))]=='X':
                    j+=1
                if j!=1:
                    if state[l[i]-(j*(m-1))]==' ' and (l[i]-(j*(m-1))) not in moves:
                        moves.append(l[i]-(j*(m-1)))
                    j=1
            return moves
        
def makeMove(state, move, player):
    '''given the state, move and player, flip all the opponents coins in all possible directions'''
    n = len(state)
    m = int(n**0.5)
    if player=='X':
        if True:         ########m==4:
            p,q = getCod(move,m)
            j=1
            #check left
            while q-j>0 and state[move-j]=='O':
                j+=1
            if j!=1:
                if state[move-j]=='X':
                    for i in range(j):
                        state[move-i]='X'
                j=1
            #check right
            while q+j<m-1 and state[move+j]=='O':
                j+=1
            if j!=1:
                if state[move+j]=='X':
                    for i in range(j):
                        state[move+i]='X'
                j=1
            #check up
            while p-j>0 and state[move-(j*m)]=='O':
                j+=1
            if j!=1:
                if state[move-(j*m)]=='X':
                    for i in range(j):
                        state[move-(i*m)]='X'
                j=1
            #check down
            while p+j<m-1 and state[move+(j*m)]=='O':
                j+=1
            if j!=1:
                if state[move+(j*m)]=='X':
                    for i in range(j):
                        state[move+(i*m)]='X'
                j=1
            #check diagonally down left 
            while (p+j<m-1 and q-j>0) and state[move+(j*(m-1))]=='O':
                j+=1
            if j!=1:
                if state[move+(j*(m-1))]=='X':
                    for i in range(j):
                        state[move+(i*(m-1))]='X'
                j=1
            #check diagonally down right 
            while (p+j<m-1 and q+j<m-1) and state[move+(j*(m+1))]=='O':
                j+=1
            if j!=1:
                if state[move+(j*(m+1))]=='X':
                    for i in range(j):
                        state[move+(i*(m+1))]='X'
                j=1
            #check diagonally up left 
            while (p-j>0 and q-j>0) and state[move-(j*(m+1))]=='O':
                j+=1
            if j!=1:
                if state[move-(j*(m+1))]=='X':
                    for i in range(j):
                        state[move-(i*(m+1))]='X'
                j=1
            #check diagonally up right 
            while (p-j>0 and q+j<m-1) and state[move-(j*(m-1))]=='O':
                j+=1
            if j!=1:
                if state[move-(j*(m-1))]=='X':
                    for i in range(j):
                        state[move-(i*(m-1))]='X'
                j=1
        return state
    else:
        if True:       ############m==4:
            p,q = getCod(move,m)
            j=1
            #check left
            while q-j>0 and state[move-j]=='X':
                j+=1
            if j!=1:
                if state[move-j]=='O':
                    for i in range(j):
                        state[move-i]='O'
                j=1
            #check right
            while q+j<m-1 and state[move+j]=='X':
                j+=1
            if j!=1:
                if state[move+j]=='O':
                    for i in range(j):
                        state[move+i]='O'
                j=1
            #check up
            while p-j>0 and state[move-(j*m)]=='X':
                j+=1
            if j!=1:
                if state[move-(j*m)]=='O':
                    for i in range(j):
                        state[move-(i*m)]='O'
                j=1
            #check down
            while p+j<m-1 and state[move+(j*m)]=='X':
                j+=1
            if j!=1:
                if state[move+(j*m)]=='O':
                    for i in range(j):
                        state[move+(i*m)]='O'
                j=1
            #check diagonally down left 
            while (p+j<m-1 and q-j>0) and state[move+(j*(m-1))]=='X':
                j+=1
            if j!=1:
                if state[move+(j*(m-1))]=='O':
                    for i in range(j):
                        state[move+(i*(m-1))]='O'
                j=1
            #check diagonally down right 
            while (p+j<m-1 and q+j<m-1) and state[move+(j*(m+1))]=='X':
                j+=1
            if j!=1:
                if state[move+(j*(m+1))]=='O':
                    for i in range(j):
                        state[move+(i*(m+1))]='O'
                j=1
            #check diagonally up left 
            while (p-j>0 and q-j>0) and state[move-(j*(m+1))]=='X':
                j+=1
            if j!=1:
                if state[move-(j*(m+1))]=='O':
                    for i in range(j):
                        state[move-(i*(m+1))]='O'
                j=1
            #check diagonally up right 
            while (p-j>0 and q+j<m-1) and state[move-(j*(m-1))]=='X':
                j+=1
            if j!=1:
                if state[move-(j*(m-1))]=='O':
                    for i in range(j):
                        state[move-(i*(m-1))]='O'
                j=1
        return state
    
def getMoves(state, player):
    moves = validMoves(state, player)
    if moves==[]:
        moves.append(-1) ###### -1 means pass the move
    return moves

###########################################################################################################################################
######## Functions related to Reinforcement Learning using Q table #####
###########################################################################################################################################

def epsilonGreedy(epsilon, Q, state):
    '''This epsilonGreedy() is for Q-table. It takes epsilon, Q, state as input and returns the best move available(it also includes -1 as move which means no move available and pass the turn)'''
    validMoves = getMoves(state, 'X')
    if np.random.uniform() < epsilon: # epsilon > .4: 
        return np.random.choice(validMoves)
    else:
        Qs = np.array([Q.get((tuple(state),m), 0) for m in validMoves])
        return validMoves[ np.argmax(Qs) ]

def trainQ(startState, maxGames, rho, epsilonDecayRate):
    '''Its train() for Q table. It takes startState, maxGames, rho, epsilonDecayFactor as input and returns Q table and list of outcomes and list of epsilons'''
    epsilon = 1.0
    
    outcomes = np.zeros(maxGames)
    epsilons = np.zeros(maxGames)
    Q = {}

    for nGames in range(maxGames):
        epsilon *= epsilonDecayRate
        epsilons[nGames] = epsilon
        step = 0
        board = copy(startState) # np.array([' '] * 9)  # empty board
        done = False

        while not done:        
            step += 1

            # X's turn
            move = epsilonGreedy(epsilon, Q, board)
            boardNew = copy(board)
            if move != -1:
                boardNew = makeMove(boardNew, move, 'X') # boardNew[move] = 'X'
            if (tuple(board),move) not in Q:
                Q[(tuple(board),move)] = 0  # initial Q value for new board,move
            
            if winner(boardNew) == 'X':
                # X won!
                Q[(tuple(board),move)] = 1
                done = True
                outcomes[nGames] = 1

            elif winner(boardNew)=='draw' or (boardNew.count('X')+boardNew.count('O')) == 16:
                # Game over. No winner.
                Q[(tuple(board),move)] = 0
                done = True
                outcomes[nGames] = 0

            else:
                # O's turn.  O is a random player!
                moveO = np.random.choice(getMoves(boardNew, 'O'))   ######### updated board
                if moveO != -1:
                    boardNew = makeMove(boardNew, moveO, 'O') # [moveO] = 'O'
                if winner(boardNew) == 'O':
                    # O won!
                    Q[(tuple(board),move)] += rho * (-1 - Q[(tuple(board),move)])
                    done = True
                    outcomes[nGames] = -1

            if step > 1:
                Q[(tuple(boardOld),moveOld)] += rho * (Q[(tuple(board),move)] - Q[(tuple(boardOld),moveOld)])

            boardOld, moveOld = copy(board), copy(move) # remember board and move to Q(board,move) can be updated after next steps
            board = boardNew
        
    return Q, outcomes, epsilons

def testQ(startState, Q, maxSteps, validMovesF, makeMoveF):
    '''testQ for Reversi. Its inputs are startState, Q, maxSteps, validMovesF, makeMoveF and returns path from startState to winning state if it exists ortherwise None'''
    state = copy(startState)
    path = [copy(state)]
    for step in range(maxSteps):
        moves = validMovesF(state,'X')
        
        qs = []
        for m in moves:
            key = (tuple(state), m)
            if key in Q:   ####  if key is not found in Q then should i add it to Q or not
                qs.append(Q[key])
        if qs != []:
            move = moves[np.argmax(qs)]
        else:
            # if the state is not in Q, then select a random available move
            move = np.random.choice(validMovesF(state, 'X'))
            print('key not found in Q, so random move selected')
        if move != -1:
            state = makeMoveF(state, move,'X')
            path.append(copy(state))
        # O's turn
        moveO = np.random.choice(validMovesF(state, 'O'))
        if moveO != -1:
            state = makeMoveF(state,moveO,'O')
            path.append(copy(state))

        if winner(state) == 'X':
            # goal found
            return path

    return None

###################################################################################################################################################  Functions related to Qnet ########
###########################################################################################################################################

import neuralnetworks as nn

def newStateRep(state):
    newrep = copy(state)
    for i in range(len(state)):
        if newrep[i]=='X':
            newrep[i] = 1
        elif newrep[i]=='O':
            newrep[i] = -1
        elif newrep[i]==' ':
            newrep[i] = 0
    return newrep    

def epsilonGreedyQnet(Qnet, state, epsilon, validMovesF):
    '''This function takes Qnet, state, epsilon, validMovesF as input and returns the best move available and the Q values'''
    moves = validMovesF(state,'X')
    if np.random.uniform() < epsilon: # random move
        move = np.random.choice(moves)
        Q = Qnet.use(np.array([newStateRep(state) + [move]])) if Qnet.Xmeans is not None else 0
    else:                           # greedy move
        qs = []
        for m in moves:
            qs.append(Qnet.use(np.array([newStateRep(state) + [m]])) if Qnet.Xmeans is not None else 0)
        move = moves[np.argmax(qs)]
        Q = np.max(qs)
    return move, Q

def trainQnet(startState, nReps, hiddenLayers, nIterations, nReplays, epsilon, epsilonDecayFactor, validMovesF, makeMoveF):
    '''This function trains the Qnet using startState, nReps, hiddenLayers, nIterations, nReplays, epsilon, epsilonDecayFactor, validMovesF, makeMoveF as input and returns Qnet, outcomes, epsilons and samples'''
    outcomes = np.zeros(nReps)
    epsilons = np.zeros(nReps)
    Qnet = nn.NeuralNetwork(len(startState)+1, hiddenLayers, 1)
    Qnet._standardizeT = lambda x: x
    Qnet._unstandardizeT = lambda x: x
    # epsilon = 1.0

    samples = []  # collect all samples for this repetition, then update the Q network at end of repetition.
    for rep in range(nReps):
        if rep > 0:
            epsilon *= epsilonDecayFactor
        epsilons[rep] = epsilon
        step = 0
        done = False

        samples = []
        samplesNextStateForReplay = []
        
        state = copy(startState)
        move, _ = epsilonGreedyQnet(Qnet, state, epsilon, validMovesF)
 
        while not done:
            step += 1
            
            # X's turn
            # Make this move to get to nextState
            
            stateNext = copy(state)
            if move != -1:
                stateNext = makeMoveF(stateNext, move, 'X')
                
            if winner(stateNext) == 'X':
                # X won
                Qnext = 0
                done = True
                outcomes[rep] = 1
                
            elif winner(stateNext) == 'draw':
                Qnext = 0
                done = True
                outcomes[rep] = 0
            
            else:
                # O's turn
                moveO = np.random.choice(validMovesF(stateNext, 'O'))   ########## updated state
                if moveO != -1:
                    stateNext = makeMoveF(stateNext, moveO, 'O')
                if winner(stateNext) == 'O':
                    Qnext = 0
                    done = True
                    outcomes[rep] = -1
            
            # Choose move from nextState
            moveNext, Qtemp = epsilonGreedyQnet(Qnet, stateNext, epsilon, validMovesF)
            if not done:
                Qnext = Qtemp
                
            if winner(stateNext)=='X':
                r=1
            elif winner(stateNext)=='draw':
                r=0
            elif winner(stateNext)=='O':
                r=-1
            else:
                r=-1
            samples.append([*newStateRep(state), move, r, Qnext])
            samplesNextStateForReplay.append([*newStateRep(stateNext), moveNext])
            
            state = copy(stateNext)
            move = copy(moveNext)
            
        samples = np.array(samples)
        X = samples[:,:len(startState)+1]
        T = samples[:,len(startState)+1:len(startState)+2] + samples[:,len(startState)+2:len(startState)+3]
        Qnet.train(X, T, nIterations, verbose=False)

        # Experience Replay: Train on recent samples with updates to Qnext.
        samplesNextStateForReplay = np.array(samplesNextStateForReplay)
        for replay in range(nReplays):
            # for sample, stateNext in zip(samples, samplesNextStateForReplay):
                # moveNext, Qnext = epsilonGreedyQnet(Qnet, stateNext, epsilon, validMovesF)
                # sample[6] = Qnext
            # print('before',samples[:5,6])
            QnextNotZero = samples[:,len(startState)+2] != 0
            samples[QnextNotZero, len(startState)+2:len(startState)+3] = Qnet.use(samplesNextStateForReplay[QnextNotZero,:])
            # print('after',samples[:5,6])
            T = samples[:,len(startState)+1:len(startState)+2] + samples[:,len(startState)+2:len(startState)+3]
            Qnet.train(X, T, nIterations, verbose=False)

    print('DONE')
    return Qnet, outcomes, epsilons, samples

def testQnet(startState, Qnet, maxSteps, validMovesF, makeMoveF):
    '''This function returns a path from startState to the winning state if it exists, otherwise None. The input parameters are startState, Qnet, maxSteps, validMovesF, makeMoveF'''
    state = copy(startState)
    path = [copy(state)]
    for step in range(maxSteps):
        move, _ = epsilonGreedyQnet(Qnet, state, 0, validMovesF)
        #else:
            # if the state is not in Q, then select a random available move
         #   move = np.random.choice(validMovesF(state, 'X'))
          #  print('key not found in Q, so random move selected')
        if move != -1:
            state = makeMoveF(state, move,'X')
            path.append(copy(state))
        # O's turn
        moveO = np.random.choice(validMovesF(state, 'O'))
        if moveO != -1:
            state = makeMoveF(state,moveO,'O')
            path.append(copy(state))

        if winner(state) == 'X':
            # goal found
            return path

    return None

################################################################################################################################################## Reinforcement Learning using Q table like notebook 15 ########
###########################################################################################################################################

def plotOutcomes(outcomes,epsilons,maxGames,nGames):
    if nGames==0:
        return
    nBins = 100
    nPer = int(maxGames/nBins)
    outcomeRows = outcomes.reshape((-1,nPer))
    outcomeRows = outcomeRows[:int(nGames/float(nPer))+1,:]
    avgs = np.mean(outcomeRows,axis=1)
    plt.subplot(3,1,1)
    xs = np.linspace(nPer,nGames,len(avgs))
    plt.plot(xs, avgs)
    plt.xlabel('Games')
    plt.ylabel('Mean of Outcomes\n(0=draw, 1=X win, -1=O win)')
    plt.title('Bins of {:d} Games'.format(nPer))
    plt.subplot(3,1,2)
    plt.plot(xs,np.sum(outcomeRows==1,axis=1),'g-',label='Wins')
    plt.plot(xs,np.sum(outcomeRows==-1,axis=1),'r-',label='Losses')
    plt.plot(xs,np.sum(outcomeRows==0,axis=1),'b-',label='Draws')
    plt.legend(loc="center")
    plt.ylabel('Number of Games\nin Bins of {:d}'.format(nPer))
    plt.subplot(3,1,3)
    plt.plot(epsilons[:nGames])
    plt.ylabel('$\epsilon$')
    
################################################################################################################################################### Reversi using NegamaxAB #########
###########################################################################################################################################

class Reversi(object): # m is the size of board i.e. for 4*4 m=4

    def __init__(self,m):
        self.board = [' ']*(m*m)
        self.board[self.getInd(int(m/2)-1,int(m/2)-1,m)] = 'O'
        self.board[self.getInd(int(m/2)-1,int(m/2),m)] = 'X'
        self.board[self.getInd(int(m/2),int(m/2)-1,m)] = 'X'
        self.board[self.getInd(int(m/2),int(m/2),m)] = 'O'
        #self.boardBeforeMove = self.board
        self.player = 'X'
        if False:
            self.board = ['X', 'X', ' ', 'X', 'O', 'O', ' ', ' ', ' ']
            self.player = 'O'
        self.playerLookAHead = self.player
        self.movesExplored = 0
        self.winningValue = 1

    def indX(self,state):
        l=[]
        for i in range(len(state)):
            if state[i]=='X':
                l.append(i)
        return l
    
    def indO(self,state):
        l=[]
        for i in range(len(state)):
            if state[i]=='O':
                l.append(i)
        return l
    
    def getCod(self,e,m):
        '''get the coordinates in 2d'''
        q = e % m
        for i in range(m):
            if e == (m*i)+q:
                p=i
        return p,q
    
    def getInd(self,p,q,m):
        return (p*m)+q
    
    #def locations(self, c):
     #   return [i for i, mark in enumerate(self.board) if mark == c]

    def validMoves(self,state, player):
        n = len(state)
        m = int(n**0.5)
        moves = []
        if player=='X':
            if True:      ##########m==4:
                l = self.indX(state)
                for i in range(len(l)):
                    p,q = self.getCod(l[i],m)
                    j=1
                    #check left
                    while q-j>0 and state[l[i]-j]=='O':
                        j+=1
                    if j!=1:
                        if state[l[i]-j]==' ' and (l[i]-j) not in moves:
                            moves.append(l[i]-j)
                        j=1
                    #check right
                    while q+j<m-1 and state[l[i]+j]=='O':
                        j+=1
                    if j!=1:
                        if state[l[i]+j]==' ' and (l[i]+j) not in moves:
                            moves.append(l[i]+j)
                        j=1
                    #check up
                    while p-j>0 and state[l[i]-(j*m)]=='O':
                        j+=1
                    if j!=1:
                        if state[l[i]-(j*m)]==' ' and (l[i]-(j*m)) not in moves:
                            moves.append(l[i]-(j*m))
                        j=1
                    #check down
                    while p+j<m-1 and state[l[i]+(j*m)]=='O':
                        j+=1
                    if j!=1:
                        if state[l[i]+(j*m)]==' ' and (l[i]+(j*m)) not in moves:
                            moves.append(l[i]+(j*m))
                        j=1
                    #check diagonally down left 
                    while (p+j<m-1 and q-j>0) and state[l[i]+(j*(m-1))]=='O':
                        j+=1
                    if j!=1:
                        if state[l[i]+(j*(m-1))]==' ' and (l[i]+(j*(m-1))) not in moves:
                            moves.append(l[i]+(j*(m-1)))
                        j=1
                    #check diagonally down right 
                    while (p+j<m-1 and q+j<m-1) and state[l[i]+(j*(m+1))]=='O':
                        j+=1
                    if j!=1:
                        if state[l[i]+(j*(m+1))]==' ' and (l[i]+(j*(m+1))) not in moves:
                            moves.append(l[i]+(j*(m+1)))
                        j=1
                    #check diagonally up left 
                    while (p-j>0 and q-j>0) and state[l[i]-(j*(m+1))]=='O':
                        j+=1
                    if j!=1:
                        if state[l[i]-(j*(m+1))]==' ' and (l[i]-(j*(m+1))) not in moves:
                            moves.append(l[i]-(j*(m+1)))
                        j=1
                    #check diagonally up right 
                    while (p-j>0 and q+j<m-1) and state[l[i]-(j*(m-1))]=='O':
                        j+=1
                    if j!=1:
                        if state[l[i]-(j*(m-1))]==' ' and (l[i]-(j*(m-1))) not in moves:
                            moves.append(l[i]-(j*(m-1)))
                        j=1
                return moves
        else:
            if True:       ############m==4:
                l = self.indO(state)
                for i in range(len(l)):
                    p,q = self.getCod(l[i],m)
                    j=1
                    #check left
                    while q-j>0 and state[l[i]-j]=='X':
                        j+=1
                    if j!=1:
                        if state[l[i]-j]==' ' and (l[i]-j) not in moves:
                            moves.append(l[i]-j)
                        j=1
                    #check right
                    while q+j<m-1 and state[l[i]+j]=='X':
                        j+=1
                    if j!=1:
                        if state[l[i]+j]==' ' and (l[i]+j) not in moves:
                            moves.append(l[i]+j)
                        j=1
                    #check up
                    while p-j>0 and state[l[i]-(j*m)]=='X':
                        j+=1
                    if j!=1:
                        if state[l[i]-(j*m)]==' ' and (l[i]-(j*m)) not in moves:
                            moves.append(l[i]-(j*m))
                        j=1
                    #check down
                    while p+j<m-1 and state[l[i]+(j*m)]=='X':
                        j+=1
                    if j!=1:
                        if state[l[i]+(j*m)]==' ' and (l[i]+(j*m)) not in moves:
                            moves.append(l[i]+(j*m))
                        j=1
                    #check diagonally down left 
                    while (p+j<m-1 and q-j>0) and state[l[i]+(j*(m-1))]=='X':
                        j+=1
                    if j!=1:
                        if state[l[i]+(j*(m-1))]==' ' and (l[i]+(j*(m-1))) not in moves:
                            moves.append(l[i]+(j*(m-1)))
                        j=1
                    #check diagonally down right 
                    while (p+j<m-1 and q+j<m-1) and state[l[i]+(j*(m+1))]=='X':
                        j+=1
                    if j!=1:
                        if state[l[i]+(j*(m+1))]==' ' and (l[i]+(j*(m+1))) not in moves:
                            moves.append(l[i]+(j*(m+1)))
                        j=1
                    #check diagonally up left 
                    while (p-j>0 and q-j>0) and state[l[i]-(j*(m+1))]=='X':
                        j+=1
                    if j!=1:
                        if state[l[i]-(j*(m+1))]==' ' and (l[i]-(j*(m+1))) not in moves:
                            moves.append(l[i]-(j*(m+1)))
                        j=1
                    #check diagonally up right 
                    while (p-j>0 and q+j<m-1) and state[l[i]-(j*(m-1))]=='X':
                        j+=1
                    if j!=1:
                        if state[l[i]-(j*(m-1))]==' ' and (l[i]-(j*(m-1))) not in moves:
                            moves.append(l[i]-(j*(m-1)))
                        j=1
                return moves
    
    def getMoves(self):
        moves = self.validMoves(self.board,self.player)
        if moves==[]:
            moves.append(-1) ###### -1 means pass the move
        return moves

    def winner(self,state):
        countX = state.count('X')
        countO  = state.count('O')
        nValidMovesX = len(validMoves(state, 'X')) if validMoves(state,'X') is not None else 0
        nValidMovesO = len(validMoves(state, 'O')) if validMoves(state,'O') is not None else 0
        if countX+countO==len(state):
            if countX>countO:
                return 'X'
            elif countX<countO:
                return 'O'
            elif countX==countO:
                return 'draw'
        elif nValidMovesX+nValidMovesO==0:
            if countX>countO:
                return 'X'
            elif countX<countO:
                return 'O'
            elif countX==countO:
                return 'draw'
    
    def getUtility(self):
        isXWon = True if self.winner(self.board)=='X' else False
        isOWon = True if self.winner(self.board)=='O' else False
        if isXWon:
            return 1 if self.playerLookAHead is 'X' else -1
        elif isOWon:
            return 1 if self.playerLookAHead is 'O' else -1
        elif self.winner(self.board)=='draw' or ' ' not in self.board:
            return 1
        else:
            return None  ########################################################## CHANGED FROM -0.1

    def isOver(self):
        return self.getUtility() is not None

    def makeMove(self, move):
        '''given the state, move and player, flip all the opponents coins in all possible directions'''
        if move != -1:
            state = self.board
            player = self.playerLookAHead
            from copy import copy
            self.boardBeforeMove = copy(state)
            n = len(state)
            m = int(n**0.5)
            if player=='X':
                if True: #######     m==4:
                    p,q = self.getCod(move,m)
                    j=1
                    #check left
                    while q-j>0 and state[move-j]=='O':
                        j+=1
                    if j!=1:
                        if state[move-j]=='X':
                            for i in range(j):
                                state[move-i]='X'
                        j=1
                    #check right
                    while q+j<m-1 and state[move+j]=='O':
                        j+=1
                    if j!=1:
                        if state[move+j]=='X':
                            for i in range(j):
                                state[move+i]='X'
                        j=1
                    #check up
                    while p-j>0 and state[move-(j*m)]=='O':
                        j+=1
                    if j!=1:
                        if state[move-(j*m)]=='X':
                            for i in range(j):
                                state[move-(i*m)]='X'
                        j=1
                    #check down
                    while p+j<m-1 and state[move+(j*m)]=='O':
                        j+=1
                    if j!=1:
                        if state[move+(j*m)]=='X':
                            for i in range(j):
                                state[move+(i*m)]='X'
                        j=1
                    #check diagonally down left 
                    while (p+j<m-1 and q-j>0) and state[move+(j*(m-1))]=='O':
                        j+=1
                    if j!=1:
                        if state[move+(j*(m-1))]=='X':
                            for i in range(j):
                                state[move+(i*(m-1))]='X'
                        j=1
                    #check diagonally down right 
                    while (p+j<m-1 and q+j<m-1) and state[move+(j*(m+1))]=='O':
                        j+=1
                    if j!=1:
                        if state[move+(j*(m+1))]=='X':
                            for i in range(j):
                                state[move+(i*(m+1))]='X'
                        j=1
                    #check diagonally up left 
                    while (p-j>0 and q-j>0) and state[move-(j*(m+1))]=='O':
                        j+=1
                    if j!=1:
                        if state[move-(j*(m+1))]=='X':
                            for i in range(j):
                                state[move-(i*(m+1))]='X'
                        j=1
                    #check diagonally up right 
                    while (p-j>0 and q+j<m-1) and state[move-(j*(m-1))]=='O':
                        j+=1
                    if j!=1:
                        if state[move-(j*(m-1))]=='X':
                            for i in range(j):
                                state[move-(i*(m-1))]='X'
                        j=1
                #return state
            else:
                if True:           ########m==4:
                    #print(move)
                    p,q = self.getCod(move,m)
                    j=1
                    #check left
                    while q-j>0 and state[move-j]=='X':
                        j+=1
                    if j!=1:
                        if state[move-j]=='O':
                            for i in range(j):
                                state[move-i]='O'
                        j=1
                    #check right
                    while q+j<m-1 and state[move+j]=='X':
                        j+=1
                    if j!=1:
                        if state[move+j]=='O':
                            for i in range(j):
                                state[move+i]='O'
                        j=1
                    #check up
                    while p-j>0 and state[move-(j*m)]=='X':
                        j+=1
                    if j!=1:
                        if state[move-(j*m)]=='O':
                            for i in range(j):
                                state[move-(i*m)]='O'
                        j=1
                    #check down
                    while p+j<m-1 and state[move+(j*m)]=='X':
                        j+=1
                    if j!=1:
                        if state[move+(j*m)]=='O':
                            for i in range(j):
                                state[move+(i*m)]='O'
                        j=1
                    #check diagonally down left 
                    while (p+j<m-1 and q-j>0) and state[move+(j*(m-1))]=='X':
                        j+=1
                    if j!=1:
                        if state[move+(j*(m-1))]=='O':
                            for i in range(j):
                                state[move+(i*(m-1))]='O'
                        j=1
                    #check diagonally down right 
                    while (p+j<m-1 and q+j<m-1) and state[move+(j*(m+1))]=='X':
                        j+=1
                    if j!=1:
                        if state[move+(j*(m+1))]=='O':
                            for i in range(j):
                                state[move+(i*(m+1))]='O'
                        j=1
                    #check diagonally up left 
                    while (p-j>0 and q-j>0) and state[move-(j*(m+1))]=='X':
                        j+=1
                    if j!=1:
                        if state[move-(j*(m+1))]=='O':
                            for i in range(j):
                                state[move-(i*(m+1))]='O'
                        j=1
                    #check diagonally up right 
                    while (p-j>0 and q+j<m-1) and state[move-(j*(m-1))]=='X':
                        j+=1
                    if j!=1:
                        if state[move-(j*(m-1))]=='O':
                            for i in range(j):
                                state[move-(i*(m-1))]='O'
                        j=1
                #return state
        self.movesExplored+=1
        self.playerLookAHead = 'X' if self.playerLookAHead == 'O' else 'O'

    
    #def makeMove(self, move):
     #   self.movesExplored+=1
      #  self.board[move] = self.playerLookAHead
       # self.playerLookAHead = 'X' if self.playerLookAHead == 'O' else 'O'

    def changePlayer(self):
        self.player = 'X' if self.player == 'O' else 'O'
        self.playerLookAHead = self.player

    def unmakeMove(self,state):
        self.board = state
        self.playerLookAHead = 'X' if self.playerLookAHead == 'O' else 'O'

    
    #def unmakeMove(self, move):
     #   self.board[move] = ' '
      #  self.playerLookAHead = 'X' if self.playerLookAHead == 'O' else 'O'

    def printState(self,state):
        n = len(state)
        m = n**0.5
        print('|',end='')
        for i in range(n):
            #print(state[i] if state[i] is not ' ' else i,end='|')
            print(state[i],end='|')
            if (i+1)%m==0 and i+1<n:
                print()
                print('|',end='')
        print()
    
    
    def __str__(self):
        #s = '{}|{}|{}\n-----\n{}|{}|{}\n-----\n{}|{}|{}'.format(*self.board)
        #return s
        self.printState(self.board)
    
    def getNumberMovesExplored(self):
        return self.movesExplored
    
    def getWinningValue(self):
        return self.winningValue
    
''' This is the modified version of the negamax given in the assignment. It includes two new parameters 'a' and 'b' which
represents alpha and beta values for implementing alpha-beta prunning.
'''
def negamaxAB(game, depthLeft, a, b):
    from copy import copy
    # If at terminal state or depth limit, return utility value and move None
    if game.isOver() or depthLeft == 0:
        return game.getUtility(), None
    # Find best move and its value from current state
    bestValue = -float('infinity')
    bestMove = None
    for move in game.getMoves():
        state = copy(game.board)
        # Apply a move to current state
        game.makeMove(move)
        # Use depth-first search to find eventual utility value and back it up.
        #  Negate it because it will come back in context of next player
        value, _ = negamaxAB(game, depthLeft-1, -b, -a) # Negating and swapping the 'a' and 'b' values
        # Remove the move from current state, to prepare for trying a different move
        game.unmakeMove(state) # removed the move parameter ## added state to solve the recurssive unmake
        if value is None:
            continue
        value = - value
        if value > bestValue:
            # Value for this move is better than moves tried so far from this state.
            bestValue = value
            bestMove = move
            a = value if value > a else a # Updating the current alpha value to the best value
        if bestValue >= b: # Prunning all the state from which we know 'X' has no chance of winning
            return bestValue, bestMove
    return bestValue, bestMove

def negamaxIDSab(game, depthLimit):
     # If at terminal state or depth limit, return utility value and move None
    if game.isOver() or depthLimit == 0:
        return game.getUtility(), None
    # Find best move and its value from current state
    bestValue = -float('infinity')
    bestMove = None
    a = -float('infinity')
    b = float('infinity')
    for depth in range(depthLimit):
        value, move = negamaxAB(game, depth, a, b)
        #print('value: ',value)
        if value is None:
            continue
        #value = -value
        if value> bestValue and value!=float('infinity'):
            bestValue = value
            bestMove = move
        if bestValue == game.getWinningValue():
            return bestValue, bestMove
    return bestValue, bestMove

def opponent(state):
    l = validMoves(state,'O')
    return l[0] if l!=[] else None

def playGameIDSab(game,opponent,depthLimit):
    printState(game.board)
    while not game.isOver():
        score,move = negamaxIDSab(game,depthLimit)
        if move == None :
            print('move is None. Stopping.')
            break
        #if move != -1:
        game.makeMove(move)
        print()
        print()    
        print('Player', game.player, 'to', move, 'for score' ,score)
        printState(game.board)
        print()
        if not game.isOver():
            game.changePlayer()
            opponentMove = opponent(game.board)
            if opponentMove != None:
                game.makeMove(opponentMove)
            print()
            print('Player', game.player, 'to', opponentMove)   ### FIXED ERROR IN THIS LINE!
            printState(game.board)
            game.changePlayer()
            
################################################################################################################################################# RL function similar to notebook 15 ########
##########################################################################################################################################
from IPython.display import display, clear_output

def RLmethod(startState, maxGames, rho, epsilonDecayRate, graphics = True, showMoves = False):
    #maxGames = 1000
    #rho = 0.2
    #epsilonDecayRate = 0.99
    epsilon = 1.0
    #graphics = True #False #True
    #showMoves = False # True  #not graphics

    outcomes = np.zeros(maxGames)
    epsilons = np.zeros(maxGames)
    Q = {}

    if graphics:
        fig = plt.figure(figsize=(10,10))

    for nGames in range(maxGames):
        epsilon *= epsilonDecayRate
        epsilons[nGames] = epsilon
        step = 0
        board = copy(startState) # np.array([' '] * 9)  # empty board
        done = False
        if showMoves:
                printState(board)
                print()

        while not done:        
            step += 1

            # X's turn
            move = epsilonGreedy(epsilon, Q, board)
            boardNew = copy(board)
            if move != -1:
                boardNew = makeMove(boardNew, move, 'X') # boardNew[move] = 'X'
            if (tuple(board),move) not in Q:
                Q[(tuple(board),move)] = 0  # initial Q value for new board,move
            if showMoves:
                printState(boardNew)
                print()

            if winner(boardNew) == 'X':
                # X won!
                if showMoves:
                    print('        X Won!')
                Q[(tuple(board),move)] = 1
                done = True
                outcomes[nGames] = 1

            elif winner(boardNew)=='draw':
                # Game over. No winner.
                if showMoves:
                    print('        draw.')
                Q[(tuple(board),move)] = 0
                done = True
                outcomes[nGames] = 0

            else:
                # O's turn.  O is a random player!
                moveO = np.random.choice(getMoves(boardNew, 'O')) ################## there was an error in this line 'board'->'boardNew'
                if moveO != -1:
                    boardNew = makeMove(boardNew, moveO, 'O') # [moveO] = 'O'
                if showMoves:
                    printState(boardNew)
                    print()
                if winner(boardNew) == 'O':
                    # O won!
                    if showMoves:
                        print('        O Won!')
                    Q[(tuple(board),move)] += rho * (-1 - Q[(tuple(board),move)])
                    done = True
                    outcomes[nGames] = -1

            if step > 1:
                Q[(tuple(boardOld),moveOld)] += rho * (Q[(tuple(board),move)] - Q[(tuple(boardOld),moveOld)])

            boardOld, moveOld = copy(board), copy(move) # remember board and move to Q(board,move) can be updated after next steps
            board = boardNew

            if graphics and (nGames % (maxGames/10) == 0 or nGames == maxGames-1):
                fig.clf() 
                plotOutcomes(outcomes,epsilons,maxGames,nGames-1)
                clear_output(wait=True)
                display(fig);

    if graphics:
        clear_output(wait=True)
    print('Outcomes: {:d} X wins {:d} O wins {:d} draws'.format(np.sum(outcomes==1), np.sum(outcomes==-1), np.sum(outcomes==0)))
    
    return Q