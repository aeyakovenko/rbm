{-|
Module      : Data.MLP
Description : Back-propagation
Copyright   : (c) Anatoly Yakovenko, 2015-2016
License     : MIT
Maintainer  : aeyakovenko@gmail.com
Stability   : experimental
Portability : POSIX

This module implements back-propagation multi-layer preceptron networks.
-}

{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Data.MLP(new
               ,feedForward
               ,backPropagate
               )where

import qualified Data.Matrix as M
import Control.Monad(foldM)
import Data.Matrix(Matrix(..)
                  ,(*^)
                  ,(+^)
                  ,(-^)
                  ,U
                  ,I
                  ,B
                  ,H
                  )

type MLP = [Matrix U I H]

-- |generate a new network from a list of layer sizes starting with the input layer
new :: Int -> [Int] -> [Matrix U I H]
new _ [] = []
new _ [_] = []
new seed (ni:nh:rest) = M.randomish (ni,nh) (-0.01,0.01) seed
                      : new (seed + 1) (nh:rest)

-- |run the feedForward algorithm over the network with the input data
feedForward :: Monad m => MLP -> Matrix U B I -> m (Matrix U B H)
feedForward nn ins = M.cast2 <$> foldM feed ins nn
   where feed a b = M.cast2 <$> feedForward1 a b
{-# INLINE feedForward #-}

-- |run the back-propagation algorithm
backPropagate :: Monad m => MLP -> Double -> Matrix U B I -> Matrix U B H -> m (MLP, Double)
backPropagate nn lc ins tbj = do
   outs <- scanForward ins nn
   let routs = map M.cast2 $ reverse outs
   let rnn = reverse nn
   let result = head routs

   -- output layer backprop
   !errm <- M.d2u $ result -^ tbj
   !odbh <- backPropOutput result errm

   -- hidden layer backprop results
   let back !delta !ons = M.cast2 <$> backPropHidden delta ons
   !rdbhs <- scanM back odbh (zip (tail routs) rnn)

   -- apply the backprops
   let dbhs = tail $ reverse rdbhs
   let inss = map M.cast2 outs
   unn <- mapM (applyBackPropH lc) (zip3 nn dbhs inss)
   err <- M.mse errm
   return (unn, err)
{-# INLINE backPropagate #-}

-- |apply backprop to the hidden nodes
applyBackPropH :: Monad m => Double -> (Matrix U I H, Matrix U B H, Matrix U B I) -> m (Matrix U I H)
applyBackPropH lc !(wij,dbh,obi) = do
   oib <- M.transpose obi
   lij <- oib `M.mmult` dbh

   -- calculate the average weight and average update
   !wave <- M.sum $ M.map abs wij
   !uave <- M.sum $ M.map abs lij
   -- scale the updates to the learning rate
   let lc' | wave > uave || uave == 0 = lc 
           | otherwise = (wave / uave) * lc 
   let uij = M.map ((*) (negate lc')) lij
   !uw <- M.d2u $ wij +^ uij
   return uw
{-# INLINE applyBackPropH #-}

backPropOutput :: Monad m => Matrix U B H -> Matrix U B H -> m (Matrix U B H)
backPropOutput obj ebj = M.d2u $ (M.map dsigmoid obj) *^ ebj 
{-# INLINE backPropOutput #-}

-- |calculate the backprop for the hidden layers
backPropHidden :: Monad m => Matrix U B H -> (Matrix U B I, Matrix U I H) -> m (Matrix U B I)
backPropHidden dbh (obi,wih) = do
   dib <- (wih `M.mmultT` dbh)
   dbi <- M.transpose dib
   M.d2u $ (M.map dsigmoid obi) *^ dbi
{-# INLINE backPropHidden #-}

-- |run the feedforward algorithm over all the layers
scanForward :: Monad m => Matrix U B I -> MLP -> m ([Matrix U B H])
scanForward ins nns = (map M.cast2) <$> scanM feed ins nns
   where feed ii nn = M.cast2 <$> feedForward1 ii nn
{-# INLINE scanForward #-}

-- |run the feedforward algorithm over 1 layer
feedForward1 :: Monad m => Matrix U B I -> Matrix U I H -> m (Matrix U B H)
feedForward1 !ibi wij = do
   sbj <- ibi `M.mmult` wij
   -- set bias output to 1
   let update _ _ 0 = 1
       update v _ _ = sigmoid v
   M.d2u $ M.traverse update sbj
{-# INLINE feedForward1 #-}

-- |monadic scan
scanM :: (Monad m) =>  (a -> b -> m a) -> a -> [b] -> m [a]
scanM _ a [] = return [a]
scanM f a ls = do
   x <- f a (head ls)
   xs <- scanM f x (tail ls)
   return (a:xs)
{-# INLINE scanM #-}

sigmoid :: Double -> Double
sigmoid d = 1 / (1 + (exp (negate d)))
{-# INLINE sigmoid #-}

dsigmoid:: Double -> Double
dsigmoid s = s * (1 - s)
{-# INLINE dsigmoid #-}


