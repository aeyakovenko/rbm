{-# LANGUAGE BangPatterns #-}
module DBN.Repa(dbn
               ,learn
               ,learnLast
               ,generate
               ,regenerate
               ,perf
               ,test
               ,DBN
               ) where


import qualified RBM.Repa as RBM
import RBM.Repa(BxI(BxI,unBxI)
               ,BxH(BxH)
               ,HxB(unHxB)
               ,IxH(IxH)
               ,RBM
               ,rbm
               )
import qualified Data.Array.Repa as R
import qualified Data.Array.Repa.Algorithms.Matrix as R
import System.Random(RandomGen
                    ,split
                    ,mkStdGen
                    ,randomRs
                    ,random
                    )
import Control.DeepSeq(deepseq)

import System.Exit (exitFailure)
import Test.QuickCheck(verboseCheckWithResult)
import Test.QuickCheck.Test(isSuccess,stdArgs,maxSuccess,maxSize)
import Data.Word(Word8)
import Control.Monad(foldM)
import Control.Monad.Identity(runIdentity)


type DBN = [RBM]

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
learn :: (Monad m)  => [RBM.Params] -> DBN -> [m BxI] -> m DBN
learn params db batches = do
   let loop !db' (pp,ix) = do
          let tdb = take ix db'
              rdb = drop ix db'
          ndb <- learnLast batches pp tdb
          return $ ndb ++ rdb
   foldM loop db $ zip params [1..]

learnLast :: (Monad m)  => [m BxI] -> RBM.Params -> DBN -> m DBN
learnLast _ _ [] = return []
learnLast batches pars [!rb] = do 
   !nrb <- RBM.learn pars rb batches
   return $ nrb : []
learnLast batches pars (!nrb:rest) = do
   let (r1:_) = splits $ mkStdGen (RBM.seed pars)
       npars = pars { RBM.seed = fst $ random r1 }
       gen rb mbxi = do
         bxi <- mbxi
         !hxb <- bxi `deepseq` RBM.hiddenProbs rb bxi
         BxI <$> (R.transpose2P $ unHxB hxb)
   let nbs = map (gen nrb) batches
   nrbms <- learnLast nbs npars rest
   return $ nrb : nrbms

{--
 - generate a batch of output
 --}
generate :: (Functor m, Monad m, RandomGen r) => r -> DBN -> BxI -> m BxH
generate _ [] pb = return $ RBM.BxH $ RBM.unBxI pb
generate rand (rb:rest) pb = do 
   hxb <- RBM.unHxB <$> RBM.hiddenProbs rb pb
   bxi <- BxI <$> (R.transpose2P hxb)
   generate rand rest bxi
{-# INLINE generate #-}

{--
 - regenerate a batch of input
 --}
regenerate :: (Functor m, Monad m, RandomGen r) => r -> DBN -> BxH -> m BxI
regenerate r dbn' bxh = regenerate' r (reverse dbn') bxh
{-# INLINE regenerate #-}

regenerate' :: (Functor m, Monad m, RandomGen r) => r -> DBN -> BxH ->  m BxI
regenerate' _ [] pb = return $ RBM.BxI $ RBM.unBxH pb
regenerate' rand (rb:rest) pb = do 
   ixh <- IxH <$> R.transpose2P (RBM.unHxI rb)
   ixb <- RBM.unIxB <$> RBM.inputProbs ixh pb
   bxh <- BxH <$> (R.transpose2P ixb)
   regenerate' rand rest bxh
{-# INLINE regenerate' #-}

splits :: RandomGen r => r -> [r]
splits rp = rc : splits rn
   where
      (rc,rn) = split rp

-- test to see if we can learn a random string
run_prop_learned :: Double -> Int -> Int -> Int -> ([Double],[Double])
run_prop_learned rate ni nd nh = runIdentity $ do
   let rb = dbn (mr 0) layers
       nmin = (fi ni) `min` (fi nh)
       nmax = (fi ni) `max` (fi nh)
       layers = (fi ni) : (take (fi nd) ((randomRs (nmin,nmax)) (mr 5))) ++ [nh]
       inputbatchL = concat $ replicate batchsz inputlst
       inputbatch = BxI $ R.fromListUnboxed (R.Z R.:. batchsz R.:. fi ni) $ inputbatchL
       inputarr = BxI $ R.fromListUnboxed (R.Z R.:. 1 R.:. fi ni) $ inputlst
       geninputs = randomRs (0::Int,1::Int) (mr 4)
       inputlst = map fromIntegral $ take (fi ni) $ 1:geninputs
       fi ww = 1 + ww
       mr i = mkStdGen (fi ni + fi nh + i)
       batchsz = 2000
       par = RBM.params { RBM.rate = rate }
   lbn <- learn (take (length rb) $ repeat par) rb [return inputbatch]
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
   runtest "notlearnred"  prop_not_learned
   runtest "learned"      prop_learned

perf :: IO ()
perf = return ()
