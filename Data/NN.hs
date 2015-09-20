{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Data.NN where
 
import qualified Data.Matrix as M
import Control.DeepSeq(NFData)
import Control.Monad(foldM)
import Data.Matrix(Matrix(..)
                  ,(*^)
                  ,(+^)
                  ,(-^)
                  ,U
                  ,D
                  )

-- symbolic data types for matrix dimentions
data I -- number of nodes in row I
data J -- number of nodes in row J
data B -- input batch size

type NN = [Matrix U I J]

feedForward :: Monad m => NN -> Matrix U B I -> m (Matrix U B J)
feedForward nn ins = M.cast2 <$> foldM feed ins nn
   where feed a b = M.cast2 <$> feedForward1 a b
{-# INLINE feedForward #-}

backProp :: Monad m => NN -> Double -> Matrix U B I -> Matrix U B J -> m NN 
backProp nn lc ins tbj = do
   outs <- scanForward ins nn
   let routsbj = map M.cast2 $ reverse outs
   let rnn = reverse nn

   --output layer backprop
   ebj <- errorMatrix (head routsbj) tbj
   pbj <- backPropOutput (head routsbj) ebj

   --hiddel layer backprop results
   let back pb ons = M.cast2 <$> backPropHidden pb ons
   pbjs <- scanM back pbj (zip (tail routsbj) rnn)

   --apply the backprops
   let fpbjs = reverse pbjs
   let inss = ins : (map M.cast2 outs)
   mapM (applyBackProp1 lc) (zip3 nn fpbjs inss)
{-# INLINE backProp #-}

mse :: Monad m => Matrix U B J -> m Double
mse errm = do 
   terr <- M.sum $ M.map (\ x -> x ** 2) errm
   return (terr/(fromIntegral $ M.elems errm))
{-# INLINE mse #-}

{--
 - compute error
 --}
errorMatrix :: Monad m => Matrix U B J -> Matrix U B J -> m (Matrix U B J)
errorMatrix obj tbj = M.d2u $ obj -^ tbj
{-# INLINE errorMatrix #-}

{--
 - apply backprop to the NN
 --}
applyBackProp1 :: Monad m => Double -> (Matrix U I J, Matrix U B J, Matrix U B I) -> m (Matrix U I J)
applyBackProp1 lc (wij,pbj,obi) = do
   oib <- M.transpose obi
   lij <- oib `M.mmult` pbj
   let sz :: Double = 1.0 / (fromIntegral $ M.elems wij)

   --calculate the average weight and average update
   wave <- ((*) sz) <$> (M.sum $ M.map abs wij)
   uave <- ((*) sz) <$> (M.sum $ M.map abs lij)
   --scale the updates to the learning rate
   let lc' = if wave > uave then lc else (wave / uave) * lc 
   let uij = M.map ((*) (negate lc')) lij
   M.d2u $ wij +^ uij
{-# INLINE applyBackProp1 #-}

backPropOutput :: Monad m => Matrix U B J -> Matrix U B J -> m (Matrix U B J)
backPropOutput obj ebj = M.d2u $ ebj *^ obj
{-# INLINE backPropOutput #-}

--calculate the  backprop for the hidden layers
backPropHidden :: Monad m => Matrix U B J -> (Matrix U B I, Matrix U I J) -> m (Matrix U B I)
backPropHidden ebj (obi,wij) = do
   ebj <- M.transpose =<< wij `M.mmultT` ebj
   let dbi = M.map dsigmoid obi
   M.d2u $ dbi *^ (M.cast2 ebj)
{-# INLINE backPropHidden #-}

scanForward :: Monad m => Matrix U B I -> NN -> m ([Matrix U B J])
scanForward ins nns = (map M.cast2) <$> scanM feed ins nns
   where feed ii nn = M.cast2 <$> feedForward1 ii nn
{-# INLINE scanForward #-}

feedForward1 :: Monad m => Matrix U B I -> Matrix U I J -> m (Matrix U B J)
feedForward1 !ibi wij = do
   sbj <- ibi `M.mmult` wij
   M.d2u $ M.map sigmoid sbj
{-# INLINE feedForward1 #-}

scanM :: (Monad m) =>  (a -> b -> m a) -> a -> [b] -> m [a]
scanM _ a [] =  return [a]
scanM f a ls = do 
   item <- f a (head ls)
   rest <- scanM f item (tail ls)
   return (a:rest)
{-# INLINE scanM #-}


sigmoid :: Double -> Double
sigmoid d = 1 / (1 + (exp (negate d)))
{-# INLINE sigmoid #-}

dsigmoid:: Double -> Double
dsigmoid s = s * (1 - s)
{-# INLINE dsigmoid #-}


