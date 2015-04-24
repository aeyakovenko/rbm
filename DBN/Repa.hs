module DBN.Repa(dbn
               ,learn
               ,generate
               ,perf
               ,test
               ,mnist
               ) where

import qualified RBM.Repa as RBM
import RBM.Repa(BxI(BxI,unBxI)
               ,HxB(HxB,unHxB)
               ,BxH(BxH,unBxH)
               ,IxH(IxH,unIxH)
               ,HxI(HxI,unHxI)
               ,IxB(IxB,unIxB)
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

import System.Exit (exitFailure)
import Test.QuickCheck(verboseCheckWithResult)
import Test.QuickCheck.Test(isSuccess,stdArgs,maxSuccess,maxSize)
import Data.Word(Word8)
import Data.Mnist(readArray)
import Control.Monad(foldM)
import Control.Monad.Identity(runIdentity)
import System.Random(RandomGen
                    ,random
                    ,randomRs
                    ,mkStdGen
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
 - generate a batch of output
 --}
generate :: (Functor m, Monad m, RandomGen r) => r -> DBN -> BxI -> m BxH
generate _ [] pb = return $ RBM.BxH $ RBM.unBxI pb

generate rand (rb:rest) pb = do 
   let (r1:rn:_) = splits rand
   hxb <- RBM.unHxB <$> RBM.generate r1 rb pb
   bxi <- BxI <$> (R.transpose2P hxb)
   generate rn rest bxi
{-# INLINE generate #-}

{--
 - regenerate a batch of input
 --}
regenerate :: (Functor m, Monad m, RandomGen r) => r -> DBN -> BxH -> m BxI
regenerate r dbn bxh = regenerate' r (reverse dbn) bxh
{-# INLINE regenerate #-}

regenerate' :: (Functor m, Monad m, RandomGen r) => r -> DBN -> BxH ->  m BxI
regenerate' _ [] pb = return $ RBM.BxI $ RBM.unBxH pb

regenerate' rand (rb:rest) pb = do 
   let (r1:rn:_) = splits rand
   ixh <- IxH <$> R.transpose2P (unHxI rb)
   ixb <- RBM.unIxB <$> RBM.regenerate r1 ixh pb
   bxh <- BxH <$> (R.transpose2P ixb)
   regenerate' rn rest bxh
{-# INLINE regenerate' #-}

{--
 - given a batch of generated hidden nodes
 - calculate the probability of each node being one
 - basically, a way to sample the probability if the batch 
 - from the same class of inputs
 --}
probV :: Monad m => HxB -> m PV 
probV hxb = do 
   sums <- R.sumP (RBM.unHxB hxb)
   let nb = R.col $ R.extent $ RBM.unHxB hxb
   R.computeUnboxedP $ R.map (prob nb) sums

prob :: Int -> Double -> Double
prob nb xx = xx / (fromIntegral nb)

splits :: RandomGen r => r -> [r]
splits rp = rc : splits rn
   where
      (rc,rn) = split rp

row :: R.DIM2 -> Int
row (R.Z R.:. r R.:. _) = r
{-# INLINE row #-}

-- test to see if we can learn a random string
run_prop_learned :: Double -> Int -> Int -> Int -> ([Double],[Double])
run_prop_learned rate ni nd nh = runIdentity $ do
   let rb = dbn (mr 0) layers
       nmin = ni `min` nh
       nmax = ni `max` nh
       layers = (fi ni) : (take (fi nd) ((randomRs (nmin,nmax)) (mr 5))) ++ [nh]
       inputbatchL = concat $ replicate batchsz inputlst
       inputbatch = BxI $ R.fromListUnboxed (R.Z R.:. batchsz R.:. fi ni) $ inputbatchL
       inputarr = BxI $ R.fromListUnboxed (R.Z R.:. 1 R.:. fi ni) $ inputlst
       geninputs = randomRs (0::Int,1::Int) (mr 4)
       inputlst = map fromIntegral $ take (fi ni) $ 1:geninputs
       fi ww = 1 + ww
       mr i = mkStdGen (fi ni + fi nh + i)
       batchsz = 2000
   lbn <- learn (mr 1) rate rb [inputbatch]
   bxh <- generate (mr 3) lbn inputarr
   bxi <- regenerate (mr 2) lbn bxh
   return $ ((tail $ R.toList $ unBxI $ bxi), (tail $ R.toList $ unBxI $ inputarr))

prop_learned :: Word8 -> Word8 -> Bool
prop_learned ni nh = (uncurry (==)) $ run_prop_learned 1.0 (fi ni) 2 (fi nh)
   where
      fi = fromIntegral

prop_not_learned :: Word8 -> Word8 -> Bool
prop_not_learned ni nh = (uncurry check) $ run_prop_learned (-1.0) (fi ni) 2 (fi nh)
   where
      fi ii = fromIntegral ii
      check aas bbs = (null aas) || (aas /= bbs) || (minimum aas == 1.0)
 
test :: IO ()
test = do
   let check rr = if (isSuccess rr) then return () else exitFailure
       cfg = stdArgs { maxSuccess = 100, maxSize = 10 }
       runtest tst p =  do putStrLn tst; check =<< verboseCheckWithResult cfg p
   runtest "learned"      prop_learned
   runtest "notlearnred"  prop_not_learned

perf :: IO ()
perf = return ()

mnist :: IO ()
mnist = do 
   let
      gen = mkStdGen 0
      ds = dbn gen [785,501,501,11]
      learnBatch :: DBN -> Int -> IO DBN
      learnBatch db ix = do
         let name = "dist/train" ++ (show ix)
         batch <- readArray name
         putStrLn $ "training: " ++ name
         let learn' dx tx = do
               dn <- learn (mkStdGen tx) 0.01 dx [(BxI batch)]
               if tx `mod` 10 == 0 then testBatch dn 0 else return ()
               return dn
         dn <- foldM learn' db [1..100]
         return dn
      testBatch :: DBN -> Int -> IO ()
      testBatch db ix = do
         let name = "dist/test" ++ (show ix)
         b <- readArray name
         bxh <- generate gen db $ BxI b
         hxb <- HxB <$> (R.transpose2P $ unBxH bxh)
         pv <- probV hxb
         print (ix, R.toList pv)
   de <- foldM learnBatch ds [0..468]
   mapM_ (testBatch de) [0..9] 
