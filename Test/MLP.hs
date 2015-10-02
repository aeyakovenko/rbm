module Test.MLP(test) where

import Data.MLP as P
import qualified Data.DNN.Trainer as T
import qualified Data.Matrix as M

--utils
import Control.Monad.Identity(runIdentity)
import Control.Monad(forever,when,foldM)

--test modules
import System.Exit (exitFailure)
import Test.QuickCheck(verboseCheckWithResult)
import Test.QuickCheck.Test(isSuccess,stdArgs,maxSuccess,maxSize)
import Data.Word(Word8)


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
       output = M.fromList (4*(fi bs),2) $ concat $ replicate (fi bs) [1,0, 1,1, 1,1, 1,0]
       fi ww = 1 + (fromIntegral ww)
       seed = fi nh
       mlp = new seed $ map fi [3,nh,2]
       train :: Monad m => T.Trainer m Bool
       train = forever $ do
         T.setLearnRate (0.001)
         ins <- mapM M.d2u $ M.splitRows (fi ri) input
         outs <- mapM M.d2u $ M.splitRows (fi ri) output
         mapM_ (uncurry T.backProp) $ zip ins outs
         gen <- T.feedForward input
         err <- M.mse (gen M.-^ output)
         when (err < 0.20) (T.finish True)
         cnt <- T.getCount
         when (cnt > 1000) (T.finish False)
   fst <$> T.run mlp train

prop_backProp1 :: Bool
prop_backProp1 = runIdentity $ do
   let input = M.fromList (1,2) [1,1]
   let output = M.fromList (1,2) [1,1]
   let mlp = M.fromList (2,2) [-1,-1,
                               -1,-1]
   let train (umlp,_) _ = P.backPropagate umlp 0.1 input output
   (_,e1) <- foldM train ([mlp],1) $ [0::Int]
   (_,e2) <- foldM train ([mlp],1) $ [0..100::Int]
   return $ e2 < e1

prop_backProp2 :: Bool
prop_backProp2 = runIdentity $ do
   let input = M.fromList (1,2) [1,1]
   let output = M.fromList (1,2) [1,0]
   let mlp = M.fromList (2,2) [1,1,
                               1,1]
   let train (umlp,_) _ = P.backPropagate umlp 0.1 input output
   (_,e1) <- foldM train ([mlp],1) $ [0::Int]
   (_,e2) <- foldM train ([mlp],1) $ [0..100::Int]
   return $ e2 < e1

prop_backProp3 :: Bool
prop_backProp3 = runIdentity $ do
   let input = M.fromList (1,2) [1,1]
   let output = M.fromList (1,2) [1,1]
   let mlp = M.fromList (2,2) [-1,-1,
                               -1,-1]
   let train (umlp,_) _ = P.backPropagate umlp 0.1 input output
   (_,e1) <- foldM train ([mlp,mlp],1) $ [0::Int]
   (_,e2) <- foldM train ([mlp,mlp],1) $ [0..100::Int]
   return $ e2 < e1

prop_backProp4 :: Bool
prop_backProp4 = runIdentity $ do
   let input = M.fromList (1,2) [1,1]
   let output = M.fromList (1,2) [1,0]
   let mlp = M.fromList (2,2) [1,1,
                               1,1]
   let train (umlp,_) _ = P.backPropagate umlp 0.1 input output
   (_,e1) <- foldM train ([mlp,mlp],1) $ [0::Int]
   (_,e2) <- foldM train ([mlp,mlp],1) $ [0..100::Int]
   return $ e2 < e1

prop_backPropXOR1 :: Bool
prop_backPropXOR1 = runIdentity $ do
   let input  = M.fromList (4,3) $ [1,1,1, 1,1,0, 1,0,1, 1,0,0]
       output = M.fromList (4,2) $ [1,0,   1,1,   1,1,   1,0]
       m1 = M.fromList (3,2) [1,1,
                              1,1,
                              1,1]
       m2 = M.fromList (2,2) [1,1,
                              1,1]
       train (umlp,_) _ = P.backPropagate umlp 0.1 input output
   (_,e1) <- foldM train ([m1,m2],1) $ [0::Int]
   (_,e2) <- foldM train ([m1,m2],1) $ [0..100::Int]
   return $ e2 < e1

prop_backPropXOR2 :: Bool
prop_backPropXOR2 = runIdentity $ do
   let input  = M.fromList (4,3) $ [1,1,1, 1,1,0, 1,0,1, 1,0,0]
       output = M.fromList (4,2) $ [1,0,   1,1,   1,1,   1,0]
       m1 = M.fromList (3,2) [-1,-1,
                              -1,-1,
                              -1,-1]
       m2 = M.fromList (2,2) [-1,-1,
                              -1,-1]
       train (umlp,_) _ = P.backPropagate umlp 0.1 input output
   (_,e1) <- foldM train ([m1,m2],1) $ [0::Int]
   (_,e2) <- foldM train ([m1,m2],1) $ [0..100::Int]
   return $ e2 < e1


prop_DxE :: Bool
prop_DxE = runIdentity $ do
   let d1 = M.fromList (2,2) [1,0,
                              0,2]
   let d2 = M.fromList (2,2) [3,0,
                              0,4]
   let d12 = M.fromList (2,2) [1,2,
                               3,4]
   let e1 = M.fromList (2,1) [5,6]
   let e2 = M.fromList (2,1) [7,8]
   let e12 = M.fromList (2,2) [5,6,
                               7,8]
   r1a <- d1 `M.mmult` e1 
   r1b <- d2 `M.mmult` e2
   r12 <- M.d2u $ d12 M.*^ e12
   return $ M.toList r1a ++ M.toList r1b == M.toList r12
   
test :: IO ()
test = do
   let check rr = if (isSuccess rr) then return () else exitFailure
       cfg = stdArgs { maxSuccess = 100, maxSize = 10 }
       runtest tst p =  do putStrLn tst; check =<< verboseCheckWithResult cfg p
   runtest "feedforward"   prop_feedForward
   runtest "feedforward1"  prop_feedForward1
   runtest "backprop1"     prop_backProp1
   runtest "backprop2"     prop_backProp2
   runtest "backprop3"     prop_backProp3
   runtest "backprop4"     prop_backProp4
   runtest "backpropXOR1"  prop_backPropXOR1
   runtest "backpropXOR2"  prop_backPropXOR2
   runtest "dxe"           prop_DxE
   runtest "backprop"      prop_backProp
 
