module Test.MLP(test) where

import Data.MLP as P
import qualified Data.DNN.Trainer as T
import qualified Data.Matrix as M

--utils
import Control.Monad.Identity(runIdentity)
import Control.Monad(forever,when)

--test modules
import System.Exit (exitFailure)
import Test.QuickCheck(verboseCheckWithResult)
import Test.QuickCheck.Test(isSuccess,stdArgs,maxSuccess,maxSize)
import Data.Word(Word8)
import Debug.Trace(traceShowId)


prop_feedForward :: Word8 -> Word8 -> Word8 -> Word8 -> Bool
prop_feedForward bs ni nh no = runIdentity $ do
   let input = M.randomish (fi bs, fi ni) (0,1) seed
       fi ww = 1 + (fromIntegral ww)
       seed = product $ map fi [bs,ni,nh,no]
       mlp = new seed $ map fi [ni,nh,no]
   outs <- P.feedForward mlp input 
   return $ (M.shape outs) == (fi bs, fi no)

sigmoid :: Double -> Double
sigmoid d = 1 / (1 + (exp (negate d)))

prop_feedForward1 :: Bool
prop_feedForward1 = runIdentity $ do
   let input = M.fromList (1,3) [1,0,1]
   let mlp = M.fromList (3,2) [1,-1,
                               1, 1,
                               1, 1]
   outs <- P.feedForward [mlp] input 
   --(1+0+1 ignored since its bias, sigmoid of -1+0+1)
   return $ (M.toList outs) == [1, sigmoid 0] 


prop_backProp :: Word8 -> Word8 -> Word8 -> Bool
prop_backProp bs ri nh = runIdentity $ do
   let input  = M.fromList (4*(fi bs),3) $ concat $ replicate (fi bs) [1,1,1, 1,1,0, 1,0,1, 1,0,0]
       output = M.fromList (4*(fi bs),1) $ concat $ replicate (fi bs) [0, 1, 1, 0]
       fi ww = 1 + (fromIntegral ww)
       seed = fi nh
       mlp = new seed $ map fi [3,nh,1]
       train :: Monad m => T.Trainer m Bool
       train = forever $ do
         T.setLearnRate (0.001)
         ins <- mapM M.d2u $ M.splitRows (fi ri) input
         outs <- mapM M.d2u $ M.splitRows (fi ri) output
         mapM_ (uncurry T.backProp) $ zip ins outs
         gen <- T.feedForward input
         err <- traceShowId <$> M.mse (gen M.-^ output)
         when (err < 0.01) (T.finish True)
         cnt <- T.getCount
         when (cnt > 10000) (T.finish False)
   fst <$> T.run mlp train

test :: IO ()
test = do
   let check rr = if (isSuccess rr) then return () else exitFailure
       cfg = stdArgs { maxSuccess = 100, maxSize = 10 }
       runtest tst p =  do putStrLn tst; check =<< verboseCheckWithResult cfg p
   runtest "feedforward"   prop_feedForward
   runtest "feedforward1"  prop_feedForward1
   runtest "backprop"      prop_backProp
 
