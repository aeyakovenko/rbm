module DBN.Repa(dbn
               ,learn
               ,generate
               ,perf
               ,test
               ) where

import qualified RBM.Repa as RBM
import RBM.Repa(BxI(BxI)
               ,BxH(unBxH)
               ,RBM
               ,rbm
               )
import qualified Data.Array.Repa as R
import qualified Data.Array.Repa.Algorithms.Matrix as R
import Control.Applicative((<$>))
import System.Random(RandomGen
                    ,split
                    )

type DBN = [RBM]
type PV = R.Array R.U R.DIM1 Double

{--
 - create a dbn with some randomized weights
 --}
dbn :: RandomGen r => r -> [Int] -> DBN
dbn _ [] = error "dbn: empty size list" 
dbn _ (_:[]) = error "dbn: needs more then 1 size" 
dbn rand (ni:nh:[]) = [rbm rand ni nh]
dbn rand (ni:nh:rest) = rbm r1 ni nh : dbn r2 (nh:rest)
   where
      (r1,r2) = split rand

{--
 - teach the dbn a batch of inputs
 --}
learn :: (Functor m, Monad m, RandomGen r) => r -> DBN -> [BxI] -> m DBN
learn _ [] _ = return []
learn rand (rb:rest) batches = do 
   let (r1:r2:rn:_) = splits rand
   nrb <- RBM.learn r1 rb batches
   nbs <- mapM (RBM.generate r2 nrb) batches
   nrbms <- learn rn rest (map (BxI . unBxH) nbs)
   return $ nrb : nrbms

{--
 - generate a probablity vector
 --}
generate :: (Functor m, Monad m, RandomGen r) => r -> DBN -> BxI -> m PV
generate _ [] pb = do
   let prob :: Double -> Double
       prob xx = xx / (fromIntegral $ row $ R.extent $ RBM.unBxI pb)
   ixb <- R.transpose2P $ RBM.unBxI pb
   sums <- R.sumP ixb
   R.computeUnboxedP $ R.map prob sums

generate rand (rb:rest) pb = do 
   let (r1:rn:_) = splits rand
   nb <- RBM.BxI <$> RBM.unBxH <$> RBM.generate r1 rb pb
   generate rn rest nb

splits :: RandomGen r => r -> [r]
splits rp = rc : splits rn
   where
      (rc,rn) = split rp

row :: R.DIM2 -> Int
row (R.Z R.:. r R.:. _) = r
{-# INLINE row #-}

test :: IO ()
test = return ()

perf :: IO ()
perf = return ()
