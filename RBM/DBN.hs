module RBM.DBN(dbn
              ,learn
              ) where

import qualified RBM.Repa as R

type DBN = [RBM]

dbn :: RandomGen r => r -> [Int] -> DBN
dbn _ [] = error "dbn: empty size list" 
dbn _ (_:[]) = error "dbn: needs more then 1 size" 
dbn rand (ni:nh) = [R.rbm rand ni nh]
dbn rand (ni:nh:rest) = R.rbm r1 ni nh : dbn r2 (nh:rest)
   where
      (r1,r2) = split rand

learn :: RandomGen r => r -> DBN -> [Array U DIM2 Double] -> DBN
learn _ [] _ = []
learn rand (rb:rest) batches = nrb : learn rn rest nb
   where
      nrb = R.learn r1 rb batches
      nbs = map (R.generate r2 nrb) batches
      (r1:r2:rn) = splits rand

outputProbs :: RandomGen r => r -> DBN -> Array U DIM2 Double -> Array U DIM1 Double
outputProbs _ [] pb = R.computeUnboxedS $ R.map prob $ R.sumP pb
   where
      prob xx = xx / (row $ R.extent pb)
outputProbs rand (rb:rest) pb = outputProbs rn rest nb
   where
      nb = R.generate r1 rb pb
      (r1:rn) = splits rand

splits :: RandomGen r => r -> [r]
splits rp = rc : splits rn
   where
      (rc,rn) = split rp

row :: DIM2 -> Int
row (Z :. r :. _) = r
{-# INLINE row #-}

col :: DIM2 -> Int
col (Z :. _ :. c) = c
{-# INLINE col #-}


