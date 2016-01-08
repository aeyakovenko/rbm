module Test.ImageUtils(test) where
import Data.ImageUtils(appendGIF, gifCat, writeBMP)
import System.IO.Unsafe(unsafePerformIO)
import qualified Data.Matrix as M

--test modules
import System.Exit (exitFailure)
import Test.QuickCheck(verboseCheckWithResult)
import Test.QuickCheck.Test(isSuccess,stdArgs,maxSuccess,maxSize)

-- |test are just to make sure things dont crash
-- |the images are verified visualy
prop_appendGIF :: Bool
prop_appendGIF = unsafePerformIO $ do
   appendGIF "dist/prop_appendGIF.gif" $ M.fromList (2,2) $ take 4 [1..]
   appendGIF "dist/prop_appendGIF.gif" $ M.fromList (2,2) $ take 4 [1..]
   return True

prop_writeBMP :: Bool
prop_writeBMP = unsafePerformIO $ do
   writeBMP "dist/prop_writeBMP.bmp" $ M.fromList (2,2) $ take 4 [1..]
   return True

prop_gifCat :: Bool
prop_gifCat = unsafePerformIO $ do
   appendGIF "dist/prop_gifCat0.gif" $ M.fromList (2,2) $ take 4 [1..]
   appendGIF "dist/prop_gifCat1.gif" $ M.fromList (2,2) $ take 4 [1..]
   gifCat "dist/prop_gifCat.gif" ["dist/prop_gifCat1.gif", "dist/prop_gifCat0.gif"]
   return True

test :: IO ()
test = do
   let check rr = if (isSuccess rr) then return () else exitFailure
       cfg = stdArgs { maxSuccess = 100, maxSize = 10 }
       runtest tst p =  do putStrLn tst; check =<< verboseCheckWithResult cfg p
   runtest "appendGif"  prop_appendGIF
   runtest "writeBMP"   prop_writeBMP
   runtest "gifCat"     prop_gifCat
 
