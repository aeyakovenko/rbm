module DBN.Repa(dbn
               ,learn
               ,generate
               ,perf
               ,test
               ) where

import qualified RBM.Repa as RBM
import RBM.Repa(BxI(BxI)
               ,HxB(unHxB)
               ,RBM
               ,rbm
               )
import qualified Data.Array.Repa as R
import qualified Data.Array.Repa.Algorithms.Matrix as R
import Control.Applicative((<$>))
import System.Random(RandomGen
                    ,split
                    ,mkStdGen
                    )
import Control.DeepSeq(deepseq)

import Data.Mnist(readArray)
import Control.Monad(foldM)

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
learn :: (Functor m, Monad m, RandomGen r) => r -> Double -> DBN -> [BxI] -> m DBN
learn _ _ [] _ = return []
learn rand rate (rb:rest) batches = do 
   let (r1:r2:rn:_) = splits rand
   nrb <- RBM.learn r1 rate rb batches
   nbs <- nrb `deepseq` (mapM (RBM.generate r2 nrb) batches)
   nbs' <- (mapM (\ bb -> BxI <$> (R.transpose2P $ unHxB bb)) nbs)
   nrbms <- learn rn rate rest nbs'
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
   hxb <- RBM.unHxB <$> RBM.generate r1 rb pb
   bxi <- BxI <$> (R.transpose2P hxb)
   generate rn rest bxi

splits :: RandomGen r => r -> [r]
splits rp = rc : splits rn
   where
      (rc,rn) = split rp

row :: R.DIM2 -> Int
row (R.Z R.:. r R.:. _) = r
{-# INLINE row #-}

perf :: IO ()
perf = return ()

test :: IO ()
test = do 
   let
      gen = mkStdGen 0
      ds = dbn gen [784,500,500,10]
      learnBatch :: DBN -> Int -> IO DBN
      learnBatch db ix = do
         let name = "dist/train" ++ (show ix)
         batch <- readArray name
         putStrLn $ "training: " ++ name
         dn <- learn (mkStdGen ix) 0.01 db [(BxI batch)]
         testBatch dn 0
         return dn
      testBatch :: DBN -> Int -> IO ()
      testBatch db ix = do
         let name = "dist/test" ++ (show ix)
         b <- readArray name
         pv <- generate gen db $ BxI b
         print (ix, R.toList pv)
   de <- foldM learnBatch ds [0,0..]
   mapM_ (testBatch de) [0..9] 
