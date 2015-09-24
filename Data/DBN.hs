{-# LANGUAGE BangPatterns #-}
module Data.DBN(dbn
               ,learn
               ,learnLast
               ,generate
               ,regenerate
               ,perf
               ,test
               ,DBN
               ) where


import qualified Data.RBM as RBM
import qualified Data.Matrix as M
import Data.Matrix(Matrix(..)
                  ,U
                  ,B
                  ,I
                  ,H
                  )
import Data.RBM(RBM
               ,rbm
               )
import System.Random(RandomGen
                    ,split
                    ,mkStdGen
                    ,randomRs
                    ,random
                    )

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
learn :: (Monad m)  => [RBM.Params] -> DBN -> [m (Matrix U B I)] -> m DBN
learn params db batches = do
   let loop !db' (pp,ix) = do
          let tdb = take ix db'
              rdb = drop ix db'
          ndb <- learnLast batches pp tdb
          return $ ndb ++ rdb
   foldM loop db $ zip params [1..]

learnLast :: (Monad m)  => [m (Matrix U B I)] -> RBM.Params -> DBN -> m DBN
learnLast _ _ [] = return []
learnLast batches pars [!rb] = do 
   !nrb <- RBM.learn pars rb batches
   return $ nrb : []
learnLast batches pars (!nrb:rest) = do
   let (r1:_) = splits $ mkStdGen (RBM.seed pars)
       npars = pars { RBM.seed = fst $ random r1 }
       gen :: Monad m => RBM.RBM -> m (Matrix U B I) -> m (Matrix U B I)
       gen rb mbxi = do
         !bxi <- mbxi
         !bxh <- RBM.hiddenProbs rb bxi
         return $ M.cast2 bxh 
   let mbxi = map (gen nrb) batches
   nrbms <- learnLast mbxi npars rest
   return $ nrb : nrbms

{--
 - generate a batch of output
 --}
generate :: (Functor m, Monad m, RandomGen r) => r -> DBN -> Matrix U B I -> m (Matrix U B H)
generate _ [] pb = return $ M.cast2 pb
generate rand (rb:rest) pb = do 
   bxh <- RBM.hiddenProbs rb pb
   let bxi = M.cast2 bxh
   generate rand rest bxi
{-# INLINE generate #-}

{--
 - regenerate a batch of input
 --}
regenerate :: (Functor m, Monad m, RandomGen r) => r -> DBN -> (Matrix U B H) -> m (Matrix U B I)
regenerate r dbn' bxh = regenerate' r (reverse dbn') bxh
{-# INLINE regenerate #-}

regenerate' :: (Functor m, Monad m, RandomGen r) => r -> DBN -> (Matrix U B H) ->  m (Matrix U B I)
regenerate' _ [] pb = return $ M.cast2 pb
regenerate' rand (ixh:rest) pb = do 
   ixb <- RBM.inputProbs ixh pb
   bxh <- M.cast2 <$> M.transpose ixb
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
       inputbatch = M.fromList (batchsz, fi ni) $ inputbatchL
       inputarr = M.fromList (1,fi ni) $ inputlst
       geninputs = randomRs (0::Int,1::Int) (mr 4)
       inputlst = map fromIntegral $ take (fi ni) (1:geninputs)
       fi ww = 1 + ww
       mr i = mkStdGen (fi ni + fi nh + i)
       batchsz = 2000
       par = RBM.params { RBM.rate = rate }
   lbn <- learn (take (length rb) $ repeat par) rb [return inputbatch]
   bxh <- generate (mr 3) lbn inputarr
   bxi <- regenerate (mr 2) lbn bxh
   return $ ((tail $ M.toList bxi), (tail $ M.toList inputarr))

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
       cfg = stdArgs { maxSuccess = 10, maxSize = 10 }
       runtest tst p =  do putStrLn tst; check =<< verboseCheckWithResult cfg p
   runtest "notlearnred"  prop_not_learned
   runtest "learned"      prop_learned

perf :: IO ()
perf = return ()
