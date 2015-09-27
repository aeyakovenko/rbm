{-# LANGUAGE BangPatterns #-}
module Data.RBM(new
               ,RBM
               ,contraDiv
               ,energy
               ,hiddenPs
               ,inputPs
               ,sample
               ,reconstruct
               ) where

import qualified System.Random as R
import qualified Data.Matrix as M
import Control.Monad(foldM)
import Data.Matrix(Matrix(..)
                  ,(*^)
                  ,(+^)
                  ,(-^)
                  ,U
                  ,I
                  ,H
                  ,B
                  )

-- | weight matrix, numInputs x numHidden
-- | where the bias node is the first node
type RBM = Matrix U I H

-- |Create an rbm with some randomized weights
new :: Int -> Int -> Int -> RBM
new seed ni nh = M.randomish (ni, nh) (-0.01, 0.01) seed

-- |Compute the energy of the RBM with the batch of input.
energy :: (Monad m) => RBM -> (Matrix U B I) -> m Double
energy rb bxi = do
   bxh <- hiddenPs rb bxi  
   ixb <- M.transpose bxi
   ixh <- ixb `M.mmult` bxh
   enr <- (M.sum $ rb *^ ixh)
   return $ negate enr

-- |Run the RBM forward
forward :: Monad m => Matrix U B I -> RBM -> m (Matrix U B I)
forward bxi rbm = M.cast2 <$> hiddenPs rbm bxi 

-- |Run the RBM backward
backward :: Monad m => Matrix U B H -> RBM -> m (Matrix U B H)
backward bxh rbm = M.cast2 <$> (M.transpose =<< inputPs rbm bxh)

-- |Reconstruct the input by folding it forward over the stack of RBMs then backwards.
reconstruct :: Monad m => Matrix U B I -> [RBM] -> m (Matrix U B I)
reconstruct ins ixhs = do 
   bxh <- M.cast2 <$> foldM forward ins ixhs
   M.cast2 <$> foldM backward bxh (reverse ixhs)

-- |Run Contrastive Divergance learning.  Return the updated RBM
contraDiv :: (Monad m) => Double -> (Matrix U I H) -> Int -> Matrix U B I -> m (Matrix U I H)
contraDiv lc ixh seed bxi = do
   !wd <- weightDiff seed ixh bxi
   !uave <- M.sum $ M.map abs wd
   !wave <- M.sum $ M.map abs ixh
   let lc' = if wave > uave || uave == 0 
             then lc 
             else (wave / uave) * lc 
       wd' = M.map ((*) lc') wd
   M.d2u $ ixh +^ wd'
{-# INLINE contraDiv #-}

weightDiff :: Monad m => Int -> Matrix U I H -> Matrix U B I -> m (Matrix U I H)
weightDiff seed ixh bxi = do
   let (s1:s2:_) = seeds seed
   ixb <- M.transpose bxi
   bxh <- sample s1 =<< hiddenPs ixh bxi 
   ixb' <- sample s2 =<< inputPs ixh bxh 
   w1 <- ixb `M.mmult` bxh
   w2 <- ixb' `M.mmult` bxh
   M.d2u $ w1 -^ w2
{-# INLINE weightDiff #-}

{-|
 - Given a biased input generate probabilities of the hidden layer
 - incuding the biased probability.
 --}
hiddenPs :: (Monad m) => RBM -> Matrix U B I -> m (Matrix U B H)
hiddenPs ixh bxi = do
   !bxh <- bxi `M.mmult` ixh 
   let update v _ c | c == 0 = 1 -- ^ set bias output to 1
                    | otherwise = sigmoid v
   M.d2u $ M.traverse update bxh
{-# INLINE hiddenPs #-}


{-|
 - Given a batch biased hidden sample generate probabilities of the input layer
 - incuding the biased probability.
 --}
inputPs :: (Monad m) => RBM -> Matrix U B H -> m (Matrix U I B)
inputPs ixh bxh = do
   !ixb <- ixh `M.mmultT` bxh
   let update v r _ | r == 0 = 1 -- ^ set bias output to 1
                    | otherwise = sigmoid v
   M.d2u $ M.traverse update ixb
{-# INLINE inputPs #-}

-- |sample the matrix of probabilities
sample :: (Monad m) => Int -> Matrix U a b -> m (Matrix U a b) 
sample seed axb = do
   let rands = M.randomish (M.shape axb) (0,1) seed 
   M.d2u $ M.zipWith checkP axb rands
{-# INLINE sample #-}

seeds :: Int -> [Int] 
seeds seed = R.randoms (R.mkStdGen seed)
{-# INLINE seeds #-}

-- | Generate a sample for each value.  If the random number is less
-- | then the value return 1, otherwise return 0.
checkP ::  Double -> Double -> Double
checkP gen rand
   | gen > rand = 1
   | otherwise = 0
{-# INLINE checkP #-}

-- | sigmoid function
sigmoid :: Double -> Double
sigmoid d = 1 / (1 + (exp (negate d)))
{-# INLINE sigmoid #-}
